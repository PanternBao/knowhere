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
#include <stdexcept>

namespace faiss {
namespace gpu {

GpuIndexIVFPQR::GpuIndexIVFPQR(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFPQR* index,
        GpuIndexIVFPQRConfig config)
        : GpuIndexIVFPQ(provider, index, config, false),
          debug_flag(config.debug_flag) {
    copyFrom(index);
}

GpuIndexIVFPQR::GpuIndexIVFPQR(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        int subQuantizers,
        int bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQRConfig config)
        : GpuIndexIVFPQ(
                  provider,
                  dims,
                  nlist,
                  subQuantizers,
                  bitsPerCode,
                  metric,
                  config),
          debug_flag(config.debug_flag) {
    verifySettings_();

    // We haven't trained ourselves, so don't construct the PQ index yet
    this->is_trained = false;
    throw std::invalid_argument("unsupported gpu train now");
}

GpuIndexIVFPQR::~GpuIndexIVFPQR() {}

void GpuIndexIVFPQR::copyFrom(const faiss::IndexIVFPQR* index) {
    //cout << "init ivfpqr" << endl;
    DeviceScope scope(config_.device);

    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();

    pq = index->pq;
    subQuantizers_ = index->pq.M;
    int refineSubQuantizers_ = index->refine_pq.M;
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
            index->refine_codes,
            refineSubQuantizers_,
            debug_flag));
    ((IVFPQR*)index_.get())->kFactor = index->k_factor;
    //    printf("init ok %f,%f",
    //           index->k_factor,
    //           index->refine_pq.centroids.data()[0]);
    // Doesn't make sense to reserve memory here
    index_->setPrecomputedCodes(usePrecomputedTables_);

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFPQR::verifySettings_() const {
    FAISS_ASSERT(
            ivfpqConfig_.indicesOptions ==INDICES_CPU ||ivfpqConfig_.indicesOptions ==INDICES_GPU_ALL);
    GpuIndexIVFPQ::verifySettings_();
    // todo:custom verifySettings
}
void GpuIndexIVFPQR::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels
        const BitsetView bitset) const {
    // Device is already set in GpuIndex::search
    FAISS_ASSERT(index_);
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<Index::idx_t, 2, true> outLabels(
            const_cast<Index::idx_t*>(labels), {n, k});

    if (bitset.empty()) {
        auto bitsetDevice = toDeviceTemporary<uint8_t, 1>(
                resources_.get(),
                config_.device,
                nullptr,
                stream,
                {0});
        ((IVFPQR*)index_.get())
                ->query(queries,
                        bitsetDevice,
                        nprobe,
                        k,
                        outDistances,
                        outLabels,
                        outLabels);
    } else {
        auto bitsetDevice = toDeviceTemporary<uint8_t, 1>(
                resources_.get(),
                config_.device,
                const_cast<uint8_t*>(bitset.data()),
                stream,
                {(int)bitset.byte_size()});
        ((IVFPQR*)index_.get())
                ->query(queries,
                        bitsetDevice,
                        nprobe,
                        k,
                        outDistances,
                        outLabels,
                        outLabels);
    }
}

} // namespace gpu
} // namespace faiss
