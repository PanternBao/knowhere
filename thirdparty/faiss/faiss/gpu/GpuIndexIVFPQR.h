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
struct GpuIndexIVFPQRConfig : public GpuIndexIVFPQConfig {};

/// IVFPQR index for the GPU
class GpuIndexIVFPQR : public GpuIndexIVFPQ {
   public:
    /// Construct from a pre-existing faiss::IndexIVFPQR instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFPQR(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFPQR* index,
            int debug_flag,
            GpuIndexIVFPQRConfig config = GpuIndexIVFPQRConfig());

    /// Construct an empty index
    GpuIndexIVFPQR(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            int subQuantizers,
            int bitsPerCode,
            faiss::MetricType metric,
            GpuIndexIVFPQRConfig config = GpuIndexIVFPQRConfig());

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
    // std::unique_ptr<RefinePQ> refine_pq;
    //    int kFactor = 4;

   public:
    int debug_flag;
};

} // namespace gpu
} // namespace faiss
