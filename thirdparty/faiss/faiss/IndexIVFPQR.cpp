/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFPQR.h>

#include <cinttypes>

#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <faiss/FaissHook.h>
#include <faiss/impl/FaissAssert.h>
#include <omp.h>
#include <iostream>
namespace faiss {

/*****************************************
 * IndexIVFPQR implementation
 ******************************************/

IndexIVFPQR::IndexIVFPQR(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        size_t M_refine,
        size_t nbits_per_idx_refine)
        : IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx),
          refine_pq(d, M_refine, nbits_per_idx_refine),
          k_factor(4) {
    by_residual = true;
}

IndexIVFPQR::IndexIVFPQR() : k_factor(1) {
    by_residual = true;
}

void IndexIVFPQR::reset() {
    IndexIVFPQ::reset();
    refine_codes.clear();
}

void IndexIVFPQR::train_residual(idx_t n, const float* x) {
    float* residual_2 = new float[n * d];
    ScopeDeleter<float> del(residual_2);
    // x：训练数据
    train_residual_o(n, x, residual_2);

    if (verbose)
        printf("training %zdx%zd 2nd level PQ quantizer on %" PRId64
               " %dD-vectors\n",
               refine_pq.M,
               refine_pq.ksub,
               n,
               d);

    refine_pq.cp.max_points_per_centroid = 1000;
    refine_pq.cp.verbose = verbose;
    // residual_2：每行的数据为："训练集和离他最近的聚类中心"的残差。
    refine_pq.train(n, residual_2);
}

void IndexIVFPQR::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    add_core(n, x, xids, nullptr);
}

void IndexIVFPQR::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx) {
    float* residual_2 = new float[n * d];
    ScopeDeleter<float> del(residual_2);

    idx_t n0 = ntotal;

    add_core_o(n, x, xids, residual_2, precomputed_idx);
    std::cout << "refine_code:" << ntotal * refine_pq.code_size << "\n";
    refine_codes.resize(ntotal * refine_pq.code_size);
    // residual_2：每行的数据为："数据和离他最近的聚类中心"的残差。
    // 这里是找到 "数据和离他最近的聚类中心"的残差进行聚类化处理。
    refine_pq.compute_codes(
            residual_2, &refine_codes[n0 * refine_pq.code_size], n);
}
#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

void IndexIVFPQR::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* idx,
        const float* L1_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats,
        const BitsetView bitset) const {
    uint64_t t0;
    TIC;
    size_t k_coarse = long(k * k_factor);
    idx_t* coarse_labels = new idx_t[k_coarse * n];
    ScopeDeleter<idx_t> del1(coarse_labels);
    { // query with quantizer levels 1 and 2.
        float* coarse_distances = new float[k_coarse * n];
        ScopeDeleter<float> del(coarse_distances);
        //搜说 k*k_factor个
        IndexIVFPQ::search_preassigned(
                n,
                x,
                k_coarse,
                idx,
                L1_dis,
                coarse_distances,
                coarse_labels,
                true,
                params,
                stats,
                bitset);
    }

    indexIVFPQ_stats.search_cycles += TOC;

    TIC;

    // 3rd level refinement
    size_t n_refine = 0;
    int nt = std::min(int(n), omp_get_max_threads());
#pragma omp parallel num_threads(nt) reduction(+ : n_refine)
    {
        // tmp buffers
        float* residual_1 = new float[2 * d];
        ScopeDeleter<float> del(residual_1);
        float* residual_2 = residual_1 + d;
#pragma omp for
        for (idx_t i = 0; i < n; i++) { // sql 数量
            const float* xq = x + i * d;
            const idx_t* shortlist = coarse_labels + k_coarse * i;
            float* heap_sim = distances + k * i;
            idx_t* heap_ids = labels + k * i;
            maxheap_heapify(k, heap_sim, heap_ids); //初始化大小为 k 的堆

            for (int j = 0; j < k_coarse; j++) { // k*k_factor
                idx_t sl = shortlist[j];

                if (sl == -1)
                    continue;

                int list_no = lo_listno(sl);
                int ofs = lo_offset(sl);
                // printf("debug_flag %d\n", debug_flag);
                if (debug_flag & PRINT_LIST_NO) {
                    if (j == k_coarse - 1) {
                        printf("list_no %d,ofs %d \n", list_no, ofs);
                    } else {
                        printf("list_no %d,ofs %d \t", list_no, ofs);
                    }
                }

                assert(list_no >= 0 && list_no < nlist);
                assert(ofs >= 0 && ofs < invlists->list_size(list_no));

                // 1st level residual ：计算query和"result所在的粗聚类"的残差
                quantizer->compute_residual(xq, residual_1, list_no);
                //                printf("coarse centroids\n");
                //                for (int l = 0; l < d; l++) {
                //                    printf("%f\t",residual_1[l]);
                //                }
                //                printf("\n");
                // 2nd level residual
                // residual_2=query和"result所在的聚类中心"的残差
                const uint8_t* l2code = invlists->get_single_code(list_no, ofs);

                pq.decode(l2code, residual_2);
                //因为第二级聚类的数据是基于第一级的残差训练而来，故要减去第一级聚类的残差。
                for (int l = 0; l < d; l++)
                    residual_2[l] = residual_1[l] - residual_2[l];
                if (debug_flag & PRINT_RESIDUAL1) {
                    printf("residual 1\n");
                    for (int l = 0; l < d; l++) {
                        printf("%f\t", residual_2[l]);
                    }
                    printf("\n");
                }
                // 3rd level residual's approximation
                // residual_1=result和"result所在的聚类中心"的残差（有损，近似值）
                idx_t id = invlists->get_single_id(list_no, ofs);
                assert(0 <= id && id < ntotal);
                refine_pq.decode(
                        &refine_codes[id * refine_pq.code_size], residual_1);
                if (debug_flag & PRINT_RESIDUAL2_CODE) {
                    printf("residual 2-id %ld\n", id);
                    for (int a1 = 0; a1 < refine_pq.code_size; a1++) {
                        printf("residual 2-code-id %d\n",
                               refine_codes[id * refine_pq.code_size + a1]);
                    }

                    printf("\n");
                }
                if (debug_flag & PRINT_RESIDUAL2) {
                    printf("residual 2\n");
                    for (int l = 0; l < d; l++) {
                        printf("%f\t", residual_1[l]);
                    }
                    printf("\n");
                }
                /*
                 * residual_1 和 residual_2都是残差，
                 * residual_1=result和"result所在的聚类中心"的残差（有损，近似值）
                 * residual_2=query和"result所在的聚类中心"的残差
                 * 由于他们都是基于同一个聚类中心的残差，不影响他俩之间的L2距离计算。
                 * 故：残差后的L2距离=残差前的L2距离=query和result之间L2距离（近似值）
                 */
                float dis = fvec_L2sqr(residual_1, residual_2, d);
                if (debug_flag & PRINT_DISTANCE) {
                    printf("l2 dis %f\t", dis);
                    if (j == k_coarse - 1) {
                        printf("\n");
                    }
                }
                //排序
                if (dis < heap_sim[0]) {
                    idx_t id_or_pair = store_pairs ? sl : id;
                    maxheap_replace_top(k, heap_sim, heap_ids, dis, id_or_pair);
                }
                n_refine++;
            }
            maxheap_reorder(k, heap_sim, heap_ids);
        }
    }
    indexIVFPQ_stats.nrefine += n_refine;
    indexIVFPQ_stats.refine_cycles += TOC;
}

void IndexIVFPQR::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    IndexIVFPQ::reconstruct_from_offset(list_no, offset, recons);

    idx_t id = invlists->get_single_id(list_no, offset);
    assert(0 <= id && id < ntotal);

    std::vector<float> r3(d);
    refine_pq.decode(&refine_codes[id * refine_pq.code_size], r3.data());
    for (int i = 0; i < d; ++i) {
        recons[i] += r3[i];
    }
}

void IndexIVFPQR::merge_from(IndexIVF& other_in, idx_t add_id) {
    IndexIVFPQR* other = dynamic_cast<IndexIVFPQR*>(&other_in);
    FAISS_THROW_IF_NOT(other);

    IndexIVF::merge_from(other_in, add_id);

    refine_codes.insert(
            refine_codes.end(),
            other->refine_codes.begin(),
            other->refine_codes.end());
    other->refine_codes.clear();
}

size_t IndexIVFPQR::remove_ids(const IDSelector& /*sel*/) {
    FAISS_THROW_MSG("not implemented");
    return 0;
}

} // namespace faiss
