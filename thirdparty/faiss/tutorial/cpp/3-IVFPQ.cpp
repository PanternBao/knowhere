/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/distances.h>
#include <cstdio>
#include <cstdlib>
#include <random>

using idx_t = faiss::Index::idx_t;

int main() {
    int d = 64;      // dimension
    int nb = 1000000; // database size
    int nq = 500;     // nb of queries

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

    int nlist = 8192;
    int k = 1000;
    int m = 64;                       // bytes per vector
    faiss::IndexFlatL2 quantizer(d); // the other index

//    {
////        quantizer.train(nb, xb);
////        quantizer.add(nb, xb);
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

    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
    int table = index.use_precomputed_table;
    index.nprobe = 512;
    index.use_precomputed_table = 1;
    index.train(nb, xb);

    //    printf("train_time%f\n",
    //    faiss::indexIVF_stats.train_q1_time.getValue()); return 0;
    faiss::indexIVF_stats.reset();
    index.add(nb, xb);

    //    { // sanity check
    //        idx_t* I = new idx_t[k * 5];
    //        float* D = new float[k * 5];
    //        index.search(5, xb, k, D, I);
    //        index.search(5, xb, k, D, I);
    //        printf("I=\n");
    //        for (int i = 0; i < 5; i++) {
    //            for (int j = 0; j < k; j++)
    //                printf("%5zd ", I[i * k + j]);
    //            printf("\n");
    //        }
    //
    //        printf("D=\n");
    //        for (int i = 0; i < 5; i++) {
    //            for (int j = 0; j < k; j++)
    //                printf("%7g ", D[i * k + j]);
    //            printf("\n");
    //        }
    //
    //        delete[] I;
    //        delete[] D;
    //    }

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        int thread=256;
        while (thread >= 2) {
            index.set_thread( thread);
            StopWatch sw = StopWatch::start();
            index.search(nq, xq, k, D, I);
            sw.stop();
            printf("%f\n", sw.getElapsedTime());
            thread=thread>>1;
        }
        delete[] I;
        delete[] D;
    }
//    { // search xq
//        idx_t* I = new idx_t[k * nq];
//        float* D = new float[k * nq];
//
//        index.nprobe = 1;
//        StopWatch sw = StopWatch::start();
//        index.search(nq, xq, k, D, I);
//        sw.stop();
//        printf("%f\n", sw.getElapsedTime());
//        delete[] I;
//        delete[] D;
//    }
//    { // search xq
//        idx_t* I = new idx_t[k * nq];
//        float* D = new float[k * nq];
//
//        index.nprobe = 1;
//        StopWatch sw = StopWatch::start();
//        index.search(nq, xq, k, D, I);
//        sw.stop();
//        printf("%f\n", sw.getElapsedTime());
//        delete[] I;
//        delete[] D;
//    }
    delete[] xb;
    delete[] xq;

    return 0;
}
