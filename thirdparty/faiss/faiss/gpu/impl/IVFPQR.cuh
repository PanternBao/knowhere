/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/CustomVariable.h>
#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/impl/RefinePQ.cuh>
#include <iostream>
using namespace std;
namespace faiss {
namespace gpu {

/// Implementing class for IVFPQR on the GPU
class IVFPQR : public IVFPQ {
   public:
    IVFPQR(GpuResources* resources,
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
           float* refinePqCentroidData,
           IndicesOptions indicesOptions,
           MemorySpace space,
           std::vector<uint8_t> refineCodes,
           int refineNumSubQuantizers,
           int debug_flag);

    ~IVFPQR() override;

    void setPrecomputedCodes(bool enable);
    virtual void query(
            Tensor<float, 2, true>& queries,
            Tensor<uint8_t, 1, true>& bitset,
            int nprobe,
            int topK,
            Tensor<float, 2, true>& outDistances,
            Tensor<Index::idx_t, 2, true>& outIndices);

   private:
    RefinePQ refinePQ;

   public:
    float kFactor = 4;
    int debug_flag = 0;
};

} // namespace gpu
} // namespace faiss
