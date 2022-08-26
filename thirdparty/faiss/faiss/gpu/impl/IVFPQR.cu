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

} // namespace gpu
} // namespace faiss
