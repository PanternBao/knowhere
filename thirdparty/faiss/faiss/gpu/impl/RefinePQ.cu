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
#include <iostream>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include "RefinePQ.cuh"

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
        std::vector<uint8_t> refineCodes,
        int debug_flag)
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
          debug_flag(debug_flag)
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
    setRefineCodes_(refineCodes);
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
        // host转换为device
        // Only needed for the duration of the transposition
        DeviceTensor<float, 3, true> pqDevice(
                resources_,
                makeTempAlloc(AllocType::Quantizer, stream),
                pqHost);
        //维度转换
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

    //    DeviceTensor<float, 3, true> pqCentroidsMiddleCode2(
    //            resources_,
    //            makeDevAlloc(AllocType::Quantizer, stream),
    //            {numSubQuantizerCodes_, numSubQuantizers_,
    //            dimPerSubQuantizer_});

    //    runTransposeAny(
    //            pqCentroidsMiddleCode_, 1, 0, pqCentroidsMiddleCode2, stream);
    //
    //    pqCentroidsMiddleCode2_ = std::move(pqCentroidsMiddleCode2);
}

void RefinePQ::setRefineCodes_(std::vector<uint8_t> refineCodes) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    nb = refineCodes.size() / numSubQuantizers_; // todo:maybe overflow

    std::cout << "nb size:" << nb << " " << refineCodes.size() << "\n";
    thrust::host_vector<uint8_t> hostMemory;

    const uint8_t* data = refineCodes.data();
    hostMemory.insert(hostMemory.end(), data, data + refineCodes.size());
    HostTensor<uint8_t, 2, true> pqHost(
            hostMemory.data(), {nb, numSubQuantizers_});

    DeviceTensor<uint8_t, 2, true> refine_code(
            resources_, makeDevAlloc(AllocType::Other, stream), pqHost);

    refineCodes_ = std::move(refine_code);
}

__global__ void getResidualVector2(
        Tensor<Index::idx_t, 2, true> outIndices, // nq * k*kFactor
        Tensor<float, 3, true> residual2,         // nq*k*kFactor*dim
        /// (sub q)(code id)(sub dim)
        Tensor<float, 3, true> pqCentroidsMiddleCode_,
        Tensor<float, 3, true> pqCentroidsMiddleCode2_,
        Tensor<uint8_t, 2, true> refineCodes_,
        int debug_flag) {
    int DimsPerSubQuantizer = pqCentroidsMiddleCode_.getSize(2);
    int numSubQuantizers_ = pqCentroidsMiddleCode_.getSize(0);
    int totalDim = residual2.getSize(2);
    int nq = outIndices.getSize(0);
    int topK = outIndices.getSize(1);
    int i = blockIdx.x;
    int j = blockIdx.y;

    Index::idx_t id = outIndices[i][j];
    if (debug_flag & PRINT_RESIDUAL2_CODE) {
        printf("residual 2-id %ld\n", id);
    }
    auto residual2Data = residual2[i][j];
    for (int currentDim = threadIdx.x; currentDim < totalDim;
         currentDim += blockDim.x) {
        int q = currentDim / DimsPerSubQuantizer;
        int l = currentDim % DimsPerSubQuantizer;
        uint8_t codeId = refineCodes_[id][q];
        if (debug_flag & PRINT_RESIDUAL2_CODE) {
            printf("residual 2-code-id %d\n", (int)codeId);
        }
        float data2 = pqCentroidsMiddleCode_[q][codeId][l];
        // float data = pqCentroidsMiddleCode2_[codeId][q][l];
        // printf("%f %f\n", data, data2);
        residual2Data[currentDim] = data2;
    }

    if (debug_flag & PRINT_RESIDUAL2) {
        printf("residual 2\n");

        for (int j = 0; j < topK; j++) {
            for (int l = 0; l < residual2.getSize(2); l++) {
                float tmp = residual2[0][j][l];
                printf("%f\t", tmp);
            }
            printf("\n");
        }

        printf("\n");
    }
}

void RefinePQ::calculateResidualVector2(
        Tensor<Index::idx_t, 2, true> outIndices, // nq * k*kFactor
        Tensor<float, 3, true> residual2          // nq*k*kFactor*dim
) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto grid = dim3(outIndices.getSize(0), outIndices.getSize(1));
    auto block = dim3(min(256, dim_));
    getResidualVector2<<<grid, block, 0, stream>>>(
            outIndices,
            residual2,
            pqCentroidsMiddleCode_,
            pqCentroidsMiddleCode2_,
            refineCodes_,
            debug_flag);
}
} // namespace gpu
} // namespace faiss
