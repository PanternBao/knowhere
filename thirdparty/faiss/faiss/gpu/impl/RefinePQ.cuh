/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/CustomVariable.h>
#include <faiss/gpu/impl/IVFPQ.cuh>
namespace faiss {
namespace gpu {

/// Implementing class for RefinePQ on the GPU
class RefinePQ : public IVFPQ {
   public:
    RefinePQ(
            GpuResources* resources,
            faiss::MetricType metric,
            float metricArg,
            /// We do not own this reference
            FlatIndex* quantizer,
            int numSubQuantizers,
            int bitsPerSubQuantizer,
            bool useFloat16LookupTables,
            bool useMMCodeDistance,
            bool interleavedLayout,
            float* pqCentroidData,
            IndicesOptions indicesOptions,
            MemorySpace space,
            std::vector<uint8_t> refineCodes,
            int debug_flag);

    ~RefinePQ() override;
    void setPQCentroids_(float* data);
    void setRefineCodes_(std::vector<uint8_t> refineCodes);
    void calculateResidualVector2(
            Tensor<Index::idx_t, 2, true> outIndices,
            Tensor<float, 3, true> residual2);

   public: // todo:change to  private
    DeviceTensor<uint8_t, 2, true> refineCodes_;
    Tensor<float, 3, true> pqCentroidsMiddleCode2_;
    int nb;
    int debug_flag=0;
    // std::vector<uint8_t> refineCodes_;
    //    /// Number of sub-quantizers per vector
    //    const int numSubQuantizers_;
    //
    //    /// Number of bits per sub-quantizer
    //    const int bitsPerSubQuantizer_;
    //
    //    /// Number of per sub-quantizer codes (2^bits)
    //    const int numSubQuantizerCodes_;
    //
    //    /// Number of dimensions per each sub-quantizer
    //    const int dimPerSubQuantizer_;
    //
    //    /// Do we maintain precomputed terms and lookup tables in float16
    //    /// form?
    //    const bool useFloat16LookupTables_;
    //
    //    /// For usage without precomputed codes, do we force usage of the
    //    /// general-purpose MM code distance computation? This is for testing
    //    /// purposes.
    //    const bool useMMCodeDistance_;
    //
    //    /// On the GPU, we prefer different PQ centroid data layouts for
    //    /// different purposes.
    //    ///
    //    /// (sub q)(sub dim)(code id)
    //    DeviceTensor<float, 3, true> pqCentroidsInnermostCode_;
    //
    //    /// (sub q)(code id)(sub dim)
    //    DeviceTensor<float, 3, true> pqCentroidsMiddleCode_;
    //
    //    /// Are precomputed codes enabled? (additional factoring and
    //    /// precomputation of the residual distance, to reduce query-time
    //    work) bool precomputedCodes_;
    //
    //    /// Precomputed term 2 in float form
    //    /// (centroid id)(sub q)(code id)
    //    DeviceTensor<float, 3, true> precomputedCode_;
    //
    //    /// Precomputed term 2 in half form
    //    DeviceTensor<half, 3, true> precomputedCodeHalf_;
};

} // namespace gpu
} // namespace faiss
