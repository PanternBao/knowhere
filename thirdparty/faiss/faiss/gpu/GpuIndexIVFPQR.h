/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/impl/IVFPQR.cuh>
#include <vector>

namespace faiss {
struct IndexIVFPQ;
}

namespace faiss {
namespace gpu {

class GpuIndexFlat;
class IVFPQ;

/// IVFPQ index for the GPU
class GpuIndexIVFPQR : public GpuIndexIVFPQ {
   public:
    /// Construct from a pre-existing faiss::IndexIVFPQR instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFPQR(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFPQR* index,
            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

    /// Construct an empty index
    GpuIndexIVFPQR(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            int subQuantizers,
            int bitsPerCode,
            faiss::MetricType metric,
            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());

    ~GpuIndexIVFPQR() override;
    void copyFrom(const faiss::IndexIVFPQR* index);
    void verifySettings_() const;


   protected:
    void searchImpl_(
            int n,
            const float* x,
            int k,
            float* distances,
            Index::idx_t* labels) const;
    std::unique_ptr<IVFPQR> refine_pq;
    int kFactor = 4;
};

} // namespace gpu
} // namespace faiss
