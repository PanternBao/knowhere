/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFPQR.h>
#include <faiss/gpu/GpuIndexIVFPQR.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFPQR::GpuIndexIVFPQR(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFPQR* index,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVFPQ(provider, index, config) {
    copyFrom(index);
}

GpuIndexIVFPQR::GpuIndexIVFPQR(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        int subQuantizers,
        int bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVFPQ(
                  provider,
                  dims,
                  nlist,
                  subQuantizers,
                  bitsPerCode,
                  metric,
                  config) {
    verifySettings_();

    // We haven't trained ourselves, so don't construct the PQ index yet
    this->is_trained = false;
}

GpuIndexIVFPQR::~GpuIndexIVFPQR() {}

void GpuIndexIVFPQR::copyFrom(const faiss::IndexIVFPQR* index) {
    DeviceScope scope(config_.device);

    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();

    pq = index->pq;
    subQuantizers_ = index->pq.M;
    bitsPerCode_ = index->pq.nbits;

    // We only support this
    FAISS_THROW_IF_NOT_MSG(
            ivfpqConfig_.interleavedLayout || index->pq.nbits == 8,
            "GPU: only pq.nbits == 8 is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->by_residual, "GPU: only by_residual = true is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->polysemous_ht == 0, "GPU: polysemous codes not supported");

    verifySettings_();

    // The other index might not be trained
    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Copy our lists as well
    // The product quantizer must have data in it
    FAISS_ASSERT(index->pq.centroids.size() > 0);
    index_.reset(new IVFPQR(
            resources_.get(),
            index->metric_type,
            index->metric_arg,
            quantizer->getGpuData(),
            subQuantizers_,
            bitsPerCode_,
            ivfpqConfig_.useFloat16LookupTables,
            ivfpqConfig_.useMMCodeDistance,
            ivfpqConfig_.interleavedLayout,
            (float*)index->pq.centroids.data(),
            (float*)index->refine_pq.centroids.data(),
            ivfpqConfig_.indicesOptions,
            config_.memorySpace,
            index->refine_codes));
    // Doesn't make sense to reserve memory here
    index_->setPrecomputedCodes(usePrecomputedTables_);

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFPQR::verifySettings_() const {
    GpuIndexIVFPQ::verifySettings_();
    // todo:custom verifySettings
}

void GpuIndexIVFPQR::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels) const {
    // Device is already set in GpuIndex::search
    FAISS_ASSERT(index_);
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
    auto stream = resources_->getDefaultStreamCurrentDevice();
    DeviceTensor<float, 2, true> tmp_distances(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k * kFactor});
    DeviceTensor<Index::idx_t, 2, true> tmp_labels(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k * kFactor});

    Tensor<float, 2, true> outDistances(tmp_distances.data(), {n, kFactor * k});
    Tensor<Index::idx_t, 2, true> outLabels(
            const_cast<Index::idx_t*>(tmp_labels.data()), {n, kFactor * k});
    index_->query(queries, nprobe, kFactor * k, outDistances, outLabels);
}

} // namespace gpu
} // namespace faiss
