/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFPQR.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/PQCodeDistances.cuh>
#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh>
#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include "IVFPQR.cuh"

namespace faiss {
namespace gpu {

IVFPQR::IVFPQR(
        GpuResources* resources,
        faiss::MetricType metric,
        float metricArg,
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
        std::vector<uint8_t> refineCodes)
        : IVFPQ(resources,
                metric,
                metricArg,
                quantizer,
                numSubQuantizers,
                bitsPerSubQuantizer,
                useFloat16LookupTables,
                useMMCodeDistance,
                interleavedLayout,
                pqCentroidData,
                indicesOptions,
                space),
          refinePQ(
                  resources,
                  metric,
                  metricArg,
                  quantizer,
                  2 * numSubQuantizers, // todo:custom numSubQuantizers
                  bitsPerSubQuantizer,
                  useFloat16LookupTables,
                  useMMCodeDistance,
                  interleavedLayout,
                  pqCentroidData,
                  indicesOptions,
                  space,refineCodes) {}

IVFPQR::~IVFPQR() {}


void IVFPQR::setPrecomputedCodes(bool enable) {
    if (enable) {
        // todo
        fprintf(stderr, "Precomputed codes is not support");
    } else {
        IVFPQ::setPrecomputedCodes(enable);
    }
}

void IVFPQR::query(
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices) {
    // These are caught at a higher level
    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    nprobe = std::min(nprobe, quantizer_->getSize());

    FAISS_ASSERT(queries.getSize(1) == dim_);
    FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
    FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

    // Reserve space for the closest coarse centroids
    // 2 代表二维数组，分配大小queries.getSize(0) * nprobe。
    //第一个模板参数是分配的类型，第二是维度
    // 具体见Tensor-inl.cuh和DeviceTensor-inl.cuh， 分配位置见AllocType
    DeviceTensor<float, 2, true> coarseDistances(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    DeviceTensor<int, 2, true> coarseIndices(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});

    //一级索引查询
    // Find the `nprobe` closest coarse centroids; we can use int
    // indices both internally and externally
    quantizer_->query(
            queries,
            nprobe,
            metric_,
            metricArg_,
            coarseDistances,
            coarseIndices,
            true);

    if (precomputedCodes_) {
        FAISS_ASSERT(metric_ == MetricType::METRIC_L2);

        runPQPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
    } else {
        runPQNoPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                k,
                outDistances,
                outIndices);
    }

    // If the GPU isn't storing indices (they are on the CPU side), we
    // need to perform the re-mapping here
    // FIXME: we might ultimately be calling this function with inputs
    // from the CPU, these are unnecessary copies
    if (indicesOptions_ == INDICES_CPU) {
        HostTensor<Index::idx_t, 2, true> hostOutIndices(outIndices, stream);

        ivfOffsetToUserIndex(
                hostOutIndices.data(),
                numLists_,
                hostOutIndices.getSize(0),
                hostOutIndices.getSize(1),
                listOffsetToUserIndex_);

        // Copy back to GPU, since the input to this function is on the
        // GPU
        outIndices.copyFrom(hostOutIndices, stream);
    }
}

} // namespace gpu
} // namespace faiss
