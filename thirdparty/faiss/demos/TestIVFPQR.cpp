/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>

#include <faiss/IndexIVFPQR.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
using idx_t = faiss::Index::idx_t;

int main() {
    setbuf(stdout, NULL);
    printf("start\n");
    const int d = 128;      // dimension
    const int nb = 100000; // database size
    const int nq = 500;     // nb of queries
    const int nlist = 32;
    const int nprobe = 1;
    const int k = 1;
    const int m = 4;
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

    faiss::IndexIVFPQR index_cpu(&quantizer, d, nlist, m, 8,m*2,8);

    index_cpu.nprobe = nprobe;
    faiss::Index* index = &index_cpu;
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
    for (int i = 0; i < 1; ++i) {
        idx_t* gt_nns = new idx_t[1 * nq];
        float* trueD = new float[1 * nq];
        faiss::IndexFlatL2 gt_index(d);
        gt_index.add(nb, xb);
        gt_index.search(nq, xq, 1, trueD, gt_nns);
        std::cout << "\n";
        //    for (int i = 0; i < 1 * nq; ++i) {
        //        std::cout << gtNNS[i] << "\t";
        //    }
        //    std::cout << "\n";
        {
            // search xq
            idx_t* nns = new idx_t[k * nq];
            float* D = new float[k * nq];
            StopWatch sw = StopWatch::start();

            index->search(nq, xq, k, D, nns);
            //        for (int i = 0; i < k * nq; ++i) {
            //            std::cout << I[i] << "\t";
            //        }

            int n_ok = 0;
            for (int q = 0; q < nq; q++) {
                for (int i = 0; i < k; i++)
                    if (nns[q * k + i] == gt_nns[q])
                        n_ok++;
            }

            std::cout << "\n";
            std::cout << n_ok << "\t" << nq;
            delete[] nns;
            delete[] D;
        }
    }

    // std::cout << ss.str();

    delete[] xb;
    delete[] xq;

    return 0;
}
