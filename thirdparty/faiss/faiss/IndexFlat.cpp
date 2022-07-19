/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include <cstring>
#include <faiss/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/StopWatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include "IndexIVFPQ.h"

using idx_t = faiss::Index::idx_t;
namespace faiss {

IndexFlat::IndexFlat(idx_t d, MetricType metric)
        : IndexFlatCodes(sizeof(float) * d, d, metric) {}
int test_search() {
    printf("start\n");
    int d = 128;     // dimension
    int nb = 100000; // database size
    int nq = 5000;   // nb of queries
    int nlist = 8192;
    int nprobe = 512;
    int k = 1000;
    int m = 64;
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

    {
        //        quantizer.train(nb, xb);
        //        quantizer.add(nb, xb);
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        StopWatch sw = StopWatch::start();
        for (int i = 0; i < 1; ++i) {
            quantizer.search2(nb, xb, k, D, I);
        }
        sw.stop();
        printf("%f\n", sw.getElapsedTime());
        return 0;
    }

    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
    int table = index.use_precomputed_table;
    index.set_thread(128);
    index.nprobe = nprobe;
    index.use_precomputed_table = 1;
    printf("train\n");
    index.train(nb, xb);

    //    printf("train_time%f\n",
    //    faiss::indexIVF_stats.train_q1_time.getValue()); return 0;
    faiss::indexIVF_stats.reset();
    printf("add\n");
    index.add(nb, xb);

    printf("search\n");
    std::stringstream ss;
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        int thread = 256;
        while (thread >= 2) {
            faiss::indexIVF_stats.reset();
            index.set_thread(thread);
            StopWatch sw = StopWatch::start();

            index.search(nq, xq, k, D, I);
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
void IndexFlat::search2(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    int loop_count = 512;
            {
                int thread = 256;
                printf("fvec_madd方法\t");
                std::function<StopWatch()> func = [&]() -> StopWatch {
                    return once_fvec_madd(n, x);
                };
                once_test(loop_count, thread, func);
            }
//    {
//        int thread = 256;
//        printf("fvec_madd2方法\t");
//        std::function<StopWatch()> func = [&]() -> StopWatch {
//            return once_fvec_madd2(n, x);
//        };
//        once_test(loop_count, thread, func);
//    }
//    {
//        int thread = 256;
//        printf("normal方法\t");
//
//        std::function<StopWatch()> func = [&]() -> StopWatch {
//            int o = 908979;
//            return once_normal(&o);
//        };
//        once_test(loop_count, thread, func);
//    }
//        {
//            int thread = 256;
//            printf("simd方法\t");
//            std::unique_ptr<float[]> x_norms(new float[n]);
//            std::function<StopWatch()> func = [&]() -> StopWatch {
//                return once_simd(n, x, x_norms.get());
//            };
//            once_test(loop_count, thread, func);
//        }
}
void IndexFlat::once_test(
        int loop_count,
        int thread,
        std::function<StopWatch()> func) const {
    double once_time = func().getElapsedTime() / 1000;
    printf("%d\t%.3f\n", loop_count, once_time);
    // printf("线程数\t总耗时\t总耗时（累加）\t平均每个线程耗时\t理想平均每个线程耗时\t每个线程耗时比\n");
    while (thread >= 4) {
        omp_set_num_threads(thread);
        StopWatch sw = StopWatch::start();
        AtomicDouble watch;

#pragma omp parallel for
        for (int i = 0; i < loop_count; ++i) {
            //#pragma omp for
            StopWatch sw2 = func();
            watch.add(sw2);
        }
        sw.stop();
        printf("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
               thread,
               sw.getElapsedTime() / 1000,
               watch.getValue(),
               watch.getValue() / thread,
               once_time * loop_count / thread,
               watch.getValue() / thread / (once_time * loop_count / thread));
        thread = thread >> 1;
    }
}
StopWatch IndexFlat::once_simd(idx_t n, const float* x, float* x_norms) const {
    StopWatch sw2 = StopWatch::start();
    for (int64_t j = 0; j < n; j++) {
        for (int i = 0; i < 1; ++i) {
            x_norms[j] = fvec_norm_L2sqr(x + j * d, d);
        }
        //
        //                        fvec_norms_L2sqr(
        //                                x_norms.get(), x, d, n); //
        //                                nx*d个 float 变为nx*1个 float
    }
    sw2.stop();
    return sw2;
}
StopWatch IndexFlat::once_fvec_madd(idx_t n, const float* x) const {
    StopWatch sw2 = StopWatch::start();
    float* D = new float[n * d];
    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 1; ++i) {
            fvec_madd(64 * 256, x, -2.0, x, D);
        }
    }
    sw2.stop();
    return sw2;
}
static void fvec_madd_ref2(
        int n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    //    for (int j = 0; j < 40; ++j) {
    //        for (long i = 0; i < n; i++)
    //            *c = c[i] + bf;
    //    }
    int n2 = 10000;
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n/n2; i++)
            c[i] = c[i] + bf;
    }
    //    for (long i = 0; i < 4000 * n; i++)
    //        *c = *c * 31;
    //        }
}
    StopWatch IndexFlat::once_fvec_madd2(idx_t n, const float* x) const {
    StopWatch sw2 = StopWatch::start();
    long n2 = 6000000;
    float* D = new float[n2];
    for (int j = 0; j < 1; ++j) {

        fvec_madd_ref2(n2, x, -2.0, x, D);
    }
    sw2.stop();
    return sw2;
}
static inline void abc(int* o, long n) {
    for (int64_t j = 0; j < n; j++) {
        *o = *o * 31;
    }
}
StopWatch IndexFlat::once_normal(int* o) const {
    StopWatch sw2 = StopWatch::start();
    int n = 4000000;
    abc(o, n);
    sw2.stop();
    return sw2;
}

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, bitset);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, bitset, train_type == 1);
    } else if (metric_type == METRIC_Jaccard) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_jaccard(x, get_xb(), d, n, ntotal, &res, bitset);
    } else {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res, bitset);
    }
}

void IndexFlat::assign(
        idx_t n,
        const float* x,
        idx_t* labels,
        float* distances) const {
    // usually used in IVF k-means algorithm
    float *dis_inner = (distances == nullptr) ? new float[n] : distances;
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
        case METRIC_L2: {
            // ignore the metric_type, both use L2
            elkan_L2_sse(x, get_xb(), d, n, ntotal, labels, dis_inner);
            break;
        }
        default: {
            // binary metrics
            // There may be something wrong, but maintain the original logic now.
            Index::assign(n, x, labels, dis_inner);
            break;
        }
    }
    if (distances == nullptr) {
        delete[] dis_inner;
    }
}

void IndexFlat::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const BitsetView bitset) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, bitset);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, bitset);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlat::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

struct FlatL2Dis : DistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float operator()(idx_t i) override {
        ndis++;
        return fvec_L2sqr(q, b + i * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlat& storage, const float* q = nullptr)
            : d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

struct FlatIPDis : DistanceComputer {
    size_t d;
    Index::idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float operator()(idx_t i) override {
        ndis++;
        return fvec_inner_product(q, b + i * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    explicit FlatIPDis(const IndexFlat& storage, const float* q = nullptr)
            : d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }
};

} // namespace

DistanceComputer* IndexFlat::get_distance_computer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
}

void IndexFlat::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlat::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

/***************************************************
 * IndexFlat1D
 ***************************************************/

IndexFlat1D::IndexFlat1D(bool continuous_update)
        : IndexFlatL2(1), continuous_update(continuous_update) {}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation() {
    perm.resize(ntotal);
    if (ntotal < 1000000) {
        fvec_argsort(ntotal, get_xb(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel(ntotal, get_xb(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add(idx_t n, const float* x) {
    IndexFlatL2::add(n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset() {
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            perm.size() == ntotal, "Call update_permutation before search");
    const float* xb = get_xb();

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        float q = x[i]; // query
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q)
                i0 = imed;
            else
                i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--;
                wp++;
                if (i0 < 0) {
                    goto finish_right;
                }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++;
                wp++;
                if (i1 >= ntotal) {
                    goto finish_left;
                }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = std::numeric_limits<float>::infinity();
                I[wp] = -1;
            }
            wp++;
        }
    done:;
    }
}

} // namespace faiss
