/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQR.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
using namespace std;
using namespace faiss;
using idx_t = Index::idx_t;
bool is_file_exist(const char* fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}
template <typename T>
void write_vector_to_file(const char* path, vector<T>& data) {
    std::ofstream ofs;
    ofs.open(path, std::ios::out | std::ios::binary);

    uint32_t sz = data.size();
    ofs.write((const char*)&sz, sizeof(uint32_t));
    for (uint32_t i = 0, end_i = data.size(); i < end_i; ++i) {
        T val = data[i];
        ofs.write((const char*)&val, sizeof(T));
    }

    ofs.close();
}

template <typename T>
vector<T> read_vector_to_file(const char* path, vector<T>& newVector) {
    std::ifstream ifs;
    ifs.open(path, std::ios::in | std::ios::binary);

    uint32_t sz = 0;
    ifs.read((char*)&sz, sizeof(uint32_t));
    ;

    for (uint32_t i = 0; i < sz; ++i) {
        T val = 0;
        ifs.read((char*)&val, sizeof(T));
        // std::cout << i << '=' << val << '\n';
        newVector[i] = val;
    }
    return newVector;
}

IndexIVFPQR* pqr_index_cpu = NULL;
IndexIVFPQ* pq_index_cpu = NULL;
const char* dataFileName = "pqr-data.dump";
const char* nqFileName = "pqr-nq.dump";
const char* pqDataFileName = "pq-data.dump";
const char* groudTruthFileName = "pqr-groudtruth.dump";
const int d = 128;      // dimension
const int nb = 1000000; // database size
const int nq = 8192;     // nb of queries
const int nlist = 8192;
const int nprobe = 156;
const int topK = 100;
const int pq_m = 128;
const int pq_refine_m = 128; // default: 2*pq_m;
const int k_factor = 4;
const int thread_num = 32;
const int debug_flag = 0;
bool needTrain = false;
int loop_count = 10;
// const int d = 128;      // dimension
// const int nb = 1000000; // database size
// const int nq = 1000;    // nb of queries
// const int nlist = 2000;
// const int nprobe = 10;
// const int topK = 10;
// const int pq_m = 32;
// const int k_factor = 4;
// const int thread_num = 8;
// const int debug_flag = 0;

bool disableIVFPQR = true;
vector<float> xq(d* nq);
vector<float> xb(d* nb);
Index* gt_index = NULL;
Index* current_index = NULL;
idx_t* gt_nns = new idx_t[1 * nq];
float* trueD = new float[1 * nq];
bool calculateGT=false;
void searchInner(int loop_count);
void search_gpu();
void search_cpu();
void calculateData();
int main() {
    // setbuf(stdout, NULL);
    printf("start GPUIVFPQR\n");
    mt19937 rng;
    uniform_real_distribution<> distrib;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    bool isFirst = !is_file_exist(pqDataFileName);
    IndexFlatL2 quantizer1(d);
    IndexFlatL2 quantizer2(d);
    if (needTrain || isFirst) {
        calculateData();

        // bytes per vector
        pqr_index_cpu =
                new IndexIVFPQR(&quantizer1, d, nlist, pq_m, 8, pq_refine_m, 8);

        //    int table = index->use_precomputed_table;
        //    index->set_thread(128);
        pqr_index_cpu->nprobe = nprobe;
        pqr_index_cpu->k_factor = k_factor;
        if (!disableIVFPQR) {
            //    index->use_precomputed_table = 1;
            printf("train\n");
            pqr_index_cpu->train(nb, xb.data());
            indexIVF_stats.reset();
            printf("add\n");
            pqr_index_cpu->add(nb, xb.data());
            fprintf(stderr, "write 1");
            write_index(pqr_index_cpu, dataFileName);
        }
        {
            // bytes per vector
            pq_index_cpu = new IndexIVFPQ(&quantizer2, d, nlist, pq_m, 8);

            //    int table = index->use_precomputed_table;
            //    index->set_thread(128);
            pq_index_cpu->nprobe = nprobe;
            //    index->use_precomputed_table = 1;
            printf("train pq index\n");
            pq_index_cpu->train(nb, xb.data());
            indexIVF_stats.reset();
            printf("add\n");
            pq_index_cpu->add(nb, xb.data());
            write_index(pq_index_cpu, pqDataFileName);
        }

        gt_index = new IndexFlatL2(d);
        gt_index->add(nb, xb.data());
        write_index(gt_index, groudTruthFileName);

        write_vector_to_file(nqFileName, xq);
        fprintf(stderr, "write done\n");
    } else {
        printf("read from existing file\n");
        // read_vector_to_file(nqFileName, xq);
        gt_index = read_index(groudTruthFileName);
        if (!disableIVFPQR) {
            pqr_index_cpu = (IndexIVFPQR*)read_index(dataFileName);
        } else {
            pqr_index_cpu =
                    new IndexIVFPQR(&quantizer1, d, nlist, pq_m, 8, pq_m, 8);
        }

        pq_index_cpu = (IndexIVFPQ*)read_index(pqDataFileName);
    }
    if (thread_num != -1) {
        pqr_index_cpu->set_thread(thread_num);
        pq_index_cpu->set_thread(thread_num);
    }
    pq_index_cpu->nprobe = nprobe;
    pqr_index_cpu->nprobe = nprobe;
    pqr_index_cpu->k_factor = k_factor;
    pqr_index_cpu->debug_flag = debug_flag;

    gt_index->search(nq, xq.data(), 1, trueD, gt_nns);
    fprintf(stderr, "calculate gt_nns done\n");
    // search_cpu();
    search_gpu();

    // std::cout << ss.str();

    delete[] gt_nns;
    delete[] trueD;
    delete gt_index;
    delete pqr_index_cpu;
    delete pq_index_cpu;

    return 0;
}
void calculateData() {
    mt19937 rng;
    uniform_real_distribution<> distrib;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }
}
void search_gpu() {
    //    int count = 10;
    //    gpu::StandardGpuResources* resArr[count];
    //    gpu::GpuIndexIVFPQ* index_gpus[count];
    //    gpu::GpuIndexIVFPQConfig config;
    //    for (int i = 0; i < count; ++i) {
    //        resArr[i] = new gpu::StandardGpuResources();
    //        index_gpus[i] = new gpu::GpuIndexIVFPQ(resArr[i], pq_index_cpu,
    //        config);
    //    }

    gpu::StandardGpuResources res; // use a single GPU

    {
        gpu::GpuIndexIVFPQConfig config;
        config.useFloat16LookupTables = true;
        config.usePrecomputedTables = true;
        config.interleavedLayout = false;
        // config.indicesOptions = gpu::IndicesOptions::INDICES_CPU;
        gpu::GpuIndexIVFPQ index_gpu_1(&res, pq_index_cpu, config);
        current_index = &index_gpu_1;
        printf("==============test_pq_gpu==============\n");
        if(disableIVFPQR){
            searchInner(loop_count);
        }else{
            searchInner(1);
        }

        printf("==============test_pq_gpu==============\n");
    }
    if (!disableIVFPQR) {
        gpu::GpuIndexIVFPQRConfig config;
        config.useFloat16LookupTables = true;
        config.usePrecomputedTables = true;
        config.indicesOptions = gpu::IndicesOptions::INDICES_GPU_ALL;
        config.debug_flag = debug_flag;
        gpu::GpuIndexIVFPQR index_gpu_2(&res, pqr_index_cpu, config);
        current_index = &index_gpu_2;
        printf("==============test_pqr_gpu==============\n");
        searchInner(loop_count);
        printf("==============test_pqr_gpu==============\n");
    }

    //    for (auto& i : resArr) {
    //        delete i;
    //    }
    //    for (auto& i : index_gpus) {
    //        delete i;
    //    }
}

void search_cpu() {
    if (!disableIVFPQR) {
        current_index = pqr_index_cpu;
        printf("==============test_pqr_cpu==============\n");
        searchInner(1);
        printf("==============test_pqr_cpu==============\n");
    }
    current_index = pq_index_cpu;
    printf("==============test_pq_cpu==============\n");
    searchInner(1);
    printf("==============test_pq_cpu==============\n");
}
void searchInner(int loop_count) {
    printf("search\n");

    for (int i = 0; i < loop_count; ++i) {
        //    for (int i = 0; i < 1 * nq; ++i) {
        //        std::cout << gtNNS[i] << "\t";
        //    }
        //    std::cout << "\n";

        {
            // search xq
            idx_t* nns = new idx_t[topK * nq];
            float* D = new float[topK * nq];
            StopWatch sw = StopWatch::start();
            printf("begin search\n");
            current_index->search(nq, xq.data(), topK, D, nns);
            //        for (int i = 0; i < topK * nq; ++i) {
            //            std::cout << I[i] << "\t";
            //        }

            sw.stop();
            cout << "take " << sw.getElapsedTime() << "ms" << endl;
            if(calculateGT) {
                int n_ok = 0;
                if (debug_flag & PRINT_MATCH_QUERY) {
                    printf("match query\n");
                }
                for (int q = 0; q < nq; q++) {
                    for (int j = 0; j < topK; j++)
                        if (nns[q * topK + j] == gt_nns[q]) {
                            if (debug_flag & PRINT_MATCH_QUERY) {
                                printf("%d,%d,%ld,%f\t",
                                       q,
                                       j,
                                       nns[q * topK + j],
                                       D[q * topK + j]);
                            }
                            n_ok++;
                            break;
                        }
                }
                cout << n_ok << "\t" << nq << "\n";
            }
            delete[] nns;
            delete[] D;
        }
    }
}
