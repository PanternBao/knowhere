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

using idx_t = faiss::Index::idx_t;
//int test_search() {
//    printf("start\n");
//    int d = 128;     // dimension
//    int nb = 100000; // database size
//    int nq = 5000;   // nb of queries
//    int nlist = 1024;
//    int nprobe = 512;
//    int k = 1000;
//    int m = 64;
//    std::mt19937 rng;
//    std::uniform_real_distribution<> distrib;
//
//    float* xb = new float[d * nb];
//    float* xq = new float[d * nq];
//
//    for (int i = 0; i < nb; i++) {
//        for (int j = 0; j < d; j++)
//            xb[d * i + j] = distrib(rng);
//        xb[d * i] += i / 1000.;
//    }
//
//    for (int i = 0; i < nq; i++) {
//        for (int j = 0; j < d; j++)
//            xq[d * i + j] = distrib(rng);
//        xq[d * i] += i / 1000.;
//    }
//
//    // bytes per vector
//    faiss::IndexFlatL2 quantizer(d); // the other index
//
//    //    {
//    //        //        quantizer.train(nb, xb);
//    //        //        quantizer.add(nb, xb);
//    //        idx_t* I = new idx_t[k * nq];
//    //        float* D = new float[k * nq];
//    //        StopWatch sw = StopWatch::start();
//    //        for (int i = 0; i < 1; ++i) {
//    //            quantizer.search2(nb, xb, k, D, I);
//    //        }
//    //        sw.stop();
//    //        printf("%f\n", sw.getElapsedTime());
//    //        return 0;
//    //    }
//
//    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
//    int table = index.use_precomputed_table;
//    index.set_thread(16);
//    index.nprobe = nprobe;
//    index.use_precomputed_table = 1;
//    printf("train\n");
//    index.train(nb, xb);
//
//    //    printf("train_time%f\n",
//    //    faiss::indexIVF_stats.train_q1_time.getValue()); return 0;
//    faiss::indexIVF_stats.reset();
//    printf("add\n");
//    index.add(nb, xb);
//
//    printf("search\n");
//    std::stringstream ss;
//    { // search xq
//        idx_t* I = new idx_t[k * nq];
//        float* D = new float[k * nq];
//        for (int i = 0; i < 10000; ++i) {
//            int thread = 16;
//            // while (thread >= 2) {
//            faiss::indexIVF_stats.reset();
//            index.set_thread(thread);
//            StopWatch sw = StopWatch::start();
//
//            index.search(nq, xq, k, D, I);
//            sw.stop();
//            ss << thread << '\t'
//               << faiss::indexIVF_stats.search_term3_add_time.getValue() << '\t'
//               << faiss::indexIVF_stats.search_lookup_time.getValue() << "\t"
//               << sw.getElapsedTime() << "\n";
//            thread = thread >> 1;
//            // }
//        }
//        delete[] I;
//        delete[] D;
//    }
//    std::cout << ss.str();
//
//    delete[] xb;
//    delete[] xq;
//
//    return 0;
//}
//
//
//void search2(
//        idx_t n,
//        const float* x,
//        idx_t k,
//        float* distances,
//        idx_t* labels) const {
//    int loop_count = 512;
//    {
//        int thread = 256;
//        printf("fvec_madd方法\t");
//        std::function<StopWatch()> func = [&]() -> StopWatch {
//            return once_fvec_madd(n, x);
//        };
//        once_test(loop_count, thread, func);
//    }
//    //    {
//    //        int thread = 256;
//    //        printf("fvec_madd2方法\t");
//    //        std::function<StopWatch()> func = [&]() -> StopWatch {
//    //            return once_fvec_madd2(n, x);
//    //        };
//    //        once_test(loop_count, thread, func);
//    //    }
//    //    {
//    //        int thread = 256;
//    //        printf("normal方法\t");
//    //
//    //        std::function<StopWatch()> func = [&]() -> StopWatch {
//    //            int o = 908979;
//    //            return once_normal(&o);
//    //        };
//    //        once_test(loop_count, thread, func);
//    //    }
//    //        {
//    //            int thread = 256;
//    //            printf("simd方法\t");
//    //            std::unique_ptr<float[]> x_norms(new float[n]);
//    //            std::function<StopWatch()> func = [&]() -> StopWatch {
//    //                return once_simd(n, x, x_norms.get());
//    //            };
//    //            once_test(loop_count, thread, func);
//    //        }
//}
//void once_test(
//        int loop_count,
//        int thread,
//        std::function<StopWatch()> func) const {
//    double once_time = func().getElapsedTime() / 1000;
//    printf("%d\t%.3f\n", loop_count, once_time);
//    // printf("线程数\t总耗时\t总耗时（累加）\t平均每个线程耗时\t理想平均每个线程耗时\t每个线程耗时比\n");
//    while (thread >= 4) {
//        omp_set_num_threads(thread);
//        StopWatch sw = StopWatch::start();
//        AtomicDouble watch;
//
//#pragma omp parallel for
//        for (int i = 0; i < loop_count; ++i) {
//            //#pragma omp for
//            StopWatch sw2 = func();
//            watch.add(sw2);
//        }
//        sw.stop();
//        printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
//               thread,
//               sw.getElapsedTime() / 1000,
//               watch.getValue(),
//               watch.getValue() / thread,
//               once_time * loop_count / thread,
//               watch.getValue() / thread / (once_time * loop_count / thread));
//        thread = thread >> 1;
//    }
//}
//StopWatch once_simd(idx_t n, const float* x, float* x_norms) const {
//    StopWatch sw2 = StopWatch::start();
//    for (int64_t j = 0; j < n; j++) {
//        for (int i = 0; i < 1; ++i) {
//            x_norms[j] = fvec_norm_L2sqr(x + j * d, d);
//        }
//        //
//        //                        fvec_norms_L2sqr(
//        //                                x_norms.get(), x, d, n); //
//        //                                nx*d个 float 变为nx*1个 float
//    }
//    sw2.stop();
//    return sw2;
//}
//StopWatch once_fvec_madd(idx_t n, const float* x) const {
//    StopWatch sw2 = StopWatch::start();
//    float* D = new float[n * d];
//    for (int j = 0; j < 20; ++j) {
//        for (int i = 0; i < 1; ++i) {
//            fvec_madd(64 * 256, x, -2.0, x, D);
//        }
//    }
//    sw2.stop();
//    return sw2;
//}
//static void fvec_madd_ref2(
//        int n,
//        const float* a,
//        float bf,
//        const float* b,
//        float* c) {
//    //    for (int j = 0; j < 40; ++j) {
//    //        for (long i = 0; i < n; i++)
//    //            *c = c[i] + bf;
//    //    }
//    int n2 = 10000;
//    for (int j = 0; j < n2; ++j) {
//        for (int i = 0; i < n / n2; i++)
//            c[i] = c[i] + bf;
//    }
//    //    for (long i = 0; i < 4000 * n; i++)
//    //        *c = *c * 31;
//    //        }
//}
//StopWatch once_fvec_madd2(idx_t n, const float* x) const {
//    StopWatch sw2 = StopWatch::start();
//    long n2 = 6000000;
//    float* D = new float[n2];
//    for (int j = 0; j < 1; ++j) {
//        fvec_madd_ref2(n2, x, -2.0, x, D);
//    }
//    sw2.stop();
//    return sw2;
//}
//static inline void abc(int* o, long n) {
//    for (int64_t j = 0; j < n; j++) {
//        *o = *o * 31;
//    }
//}
//StopWatch once_normal(int* o) const {
//    StopWatch sw2 = StopWatch::start();
//    int n = 4000000;
//    abc(o, n);
//    sw2.stop();
//    return sw2;
//}
int main() {
    return 0;
   // return test_search();
}

