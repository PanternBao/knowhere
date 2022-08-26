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
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/PQCodeDistances.cuh>
#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh>
#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
#include <faiss/gpu/impl/RefinePQ.cuh>
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

namespace faiss {
namespace gpu {

RefinePQ::RefinePQ(
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
                space)
//          ,
//          refineCodes_(refineCodes),
//          numSubQuantizers_(numSubQuantizers),
//          bitsPerSubQuantizer_(bitsPerSubQuantizer),
//          numSubQuantizerCodes_(utils::pow2(bitsPerSubQuantizer_)),
//          dimPerSubQuantizer_(dim_ / numSubQuantizers),
//          useFloat16LookupTables_(useFloat16LookupTables),
//          useMMCodeDistance_(useMMCodeDistance),
//          precomputedCodes_(false)
{
    setPQCentroids_(pqCentroidData);
}

RefinePQ::~RefinePQ() {}

void RefinePQ::setPQCentroids_(float* data) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    size_t pqSize =
            numSubQuantizers_ * numSubQuantizerCodes_ * dimPerSubQuantizer_;

    // Make sure the data is on the host
    // FIXME: why are we doing this?
    thrust::host_vector<float> hostMemory;
    hostMemory.insert(hostMemory.end(), data, data + pqSize);

    HostTensor<float, 3, true> pqHost(
            hostMemory.data(),
            {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});

    DeviceTensor<float, 3, true> pqDeviceTranspose(
            resources_,
            makeDevAlloc(AllocType::Quantizer, stream),
            {numSubQuantizers_, dimPerSubQuantizer_, numSubQuantizerCodes_});

    {
        // Only needed for the duration of the transposition
        DeviceTensor<float, 3, true> pqDevice(
                resources_,
                makeTempAlloc(AllocType::Quantizer, stream),
                pqHost);

        runTransposeAny(pqDevice, 1, 2, pqDeviceTranspose, stream);
    }

    pqCentroidsInnermostCode_ = std::move(pqDeviceTranspose);

    // Also maintain the PQ centroids in the form
    // (sub q)(code id)(sub dim)
    DeviceTensor<float, 3, true> pqCentroidsMiddleCode(
            resources_,
            makeDevAlloc(AllocType::Quantizer, stream),
            {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});

    runTransposeAny(
            pqCentroidsInnermostCode_, 1, 2, pqCentroidsMiddleCode, stream);

    pqCentroidsMiddleCode_ = std::move(pqCentroidsMiddleCode);
}

} // namespace gpu
} // namespace faiss
