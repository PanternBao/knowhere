/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/impl/RefinePQ.cuh>

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
           std::vector<uint8_t> refineCodes);

    ~IVFPQR() override;

    void setPrecomputedCodes(bool enable);
   private:
    RefinePQ refinePQ;

    /// factor between k requested in search and the k requested from the IVFPQ
    float kFactor;
};

} // namespace gpu
} // namespace faiss
