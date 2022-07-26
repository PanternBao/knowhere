/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/distances.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
using idx_t = faiss::Index::idx_t;

int main() {
    printf("start\n");
    int d = 128;     // dimension
    int nb = 1000; // database size
    int nq = 500;   // nb of queries
    int nlist = 1;
    int nprobe = 1;
    int k = 1000;
    int m = 128;
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    // bytes per vector
    faiss::IndexFlatL2 quantizer(d); // the other index

//    {
//        //        quantizer.train(nb, xb);
//        //        quantizer.add(nb, xb);
//        idx_t* I = new idx_t[k * nq];
//        float* D = new float[k * nq];
//        StopWatch sw = StopWatch::start();
//        for (int i = 0; i < 1; ++i) {
//            quantizer.search2(nb, xb, k, D, I);
//        }
//        sw.stop();
//        printf("%f\n", sw.getElapsedTime());
//        return 0;
//    }

    faiss::IndexIVFPQ index_cpu(&quantizer, d, nlist, m, 8);
    faiss::gpu::GpuClonerOptions gpu_config;
    gpu_config.useFloat16 = true;
    gpu_config.usePrecomputed =true;
    faiss::gpu::StandardGpuResources res;  // use a single GPU
    faiss::Index* index = faiss::gpu::index_cpu_to_gpu(&res, 0, &index_cpu, &gpu_config);
//    int table = index->use_precomputed_table;
//    index->set_thread(128);
//    index->nprobe = nprobe;
//    index->use_precomputed_table = 1;
    printf("train\n");
    index->train(nb, xb);

    //    printf("train_time%f\n",
    //    faiss::indexIVF_stats.train_q1_time.getValue()); return 0;
    faiss::indexIVF_stats.reset();
    printf("add\n");
    index->add(nb, xb);

    printf("search\n");
    std::stringstream ss;
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        int thread = 256;
        while (thread >= 2) {
            faiss::indexIVF_stats.reset();
            //index->set_thread(thread);
            StopWatch sw = StopWatch::start();

            index->search(nq, xq, k, D, I);
            sw.stop();
            ss << thread << '\t'
               << faiss::indexIVF_stats.search_term3_add_time.getValue() << '\t'
               << faiss::indexIVF_stats.search_lookup_time.getValue() << "\t"
               << sw.getElapsedTime() << "\n";
            thread = thread >> 1;
        }
        delete[] I;
        delete[] D;
    }
    std::cout << ss.str();

    delete[] xb;
    delete[] xq;

    return 0;
}
