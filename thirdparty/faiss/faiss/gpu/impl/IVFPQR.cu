/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda.h>
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
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "IVFPQR.cuh"

using namespace std;
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
        std::vector<uint8_t> refineCodes,
        int refineNumSubQuantizers,
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
          refinePQ(
                  resources,
                  metric,
                  metricArg,
                  quantizer, // don't use ！
                  refineNumSubQuantizers,
                  bitsPerSubQuantizer,
                  useFloat16LookupTables,
                  useMMCodeDistance,
                  interleavedLayout,
                  refinePqCentroidData,
                  indicesOptions,
                  space,
                  refineCodes,
                  debug_flag),
          debug_flag(debug_flag) {}

IVFPQR::~IVFPQR() {}

__global__ void runPQResidualVector1(
        Tensor<float, 3, true> residual1,
        Tensor<float, 2, true> queries,
        Tensor<int, 2, true> listIds,
        Tensor<int, 2, true> listOffsets,
        void** listCodes,
        /// (sub q)(sub dim)(code id)
        Tensor<float, 3, true> pqCentroidsInnermostCode_,
        /// (sub q)(code id)(sub dim)
        Tensor<float, 3, true> pqCentroidsMiddleCode_,
        Tensor<float, 3, true> listCoarseCentroids,
        int debug_flag) {
    int DimsPerSubQuantizer = pqCentroidsMiddleCode_.getSize(2);
    int numSubQuantizers_ = pqCentroidsMiddleCode_.getSize(0);
    int dim = DimsPerSubQuantizer * numSubQuantizers_;
    int nq = listIds.getSize(0);
    int topK = listIds.getSize(1);
    int i = blockIdx.x;
    auto queryData = queries[i];
    int j = blockIdx.y;
    int listId = listIds[i][j];
    int listOffset = listOffsets[i][j];
    auto coarseCentroid = listCoarseCentroids[i][j];
    if (listId == -1) {
        printf("listid is -1\n");
        return;
    }
    for (int currentDim = threadIdx.x; currentDim < dim;
         currentDim += blockDim.x) {
        int q = currentDim / DimsPerSubQuantizer;
        int l = currentDim % DimsPerSubQuantizer;

        uint8_t codeId = ((
                uint8_t*)listCodes[listId])[listOffset * numSubQuantizers_ + q];

        // printf("%d\t", i * 128 + pq_m * DimsPerSubQuantizer
        // +l);
        // int currentDim = q * DimsPerSubQuantizer + l;
        //                    printf("%f,%f,%f\n",
        //                           (float)queryData[currentDim],
        //                           (float)coarseCentroid[currentDim],
        //                           (float)pqCentroidsMiddleCode_[q][codeId][l]);
        residual1[i][j][currentDim] = queryData[currentDim] -
                coarseCentroid[currentDim] -
                pqCentroidsMiddleCode_[q][codeId][l];
    }

    if (debug_flag & PRINT_RESIDUAL1) {
        printf("residual 1\n");
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < topK; j++) {
                for (int k = 0; k < dim; k++) {
                    printf("%f\t", (float)residual1[i][j][k]);
                }
            }
            printf("\n");
        }
    }
}

__global__ void calculateListId(
        Tensor<int, 2, true> listIds,
        Tensor<int, 2, true> listOffsets,
        Tensor<Index::idx_t, 2, true> tmpOutIndices,
        int debug_flag) {
    int nq = listIds.getSize(0);
    int topK = listIds.getSize(1);
    // printf("topK,%d,%d\n",nq,topK);
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < topK; j += blockDim.x) {
        Index::idx_t sl = tmpOutIndices[i][j];

        int list_no = sl >> 32;
        int list_offset = sl & 0xffffffff;
        if (sl == -1) {
            list_no = list_offset = -1;
            printf("list_no error!");
            asm("trap;");
            return;
        }
        listOffsets[i][j] = list_offset;

        if (debug_flag & PRINT_LIST_NO) {
            printf("list_no %d,ofs %d \t", list_no, list_offset);
        }
        listIds[i][j] = list_no;
        // printf("list_no %d\t",(int) listIds[i][j]);
    }
    if (debug_flag & PRINT_LIST_NO) {
        printf("\n");
    }
}

// todo __launch_bounds__(288, 3)
__global__ void pqCodeDistances(
        Tensor<float, 3, true> residual1,
        Tensor<float, 3, true> residual2,
        Tensor<float, 2, true> outCodeDistances,
        int debug_flag) {
    int i = blockIdx.x;
    int topK = residual2.getSize(1);
    int dim = residual1.getSize(2);
    for (int j = threadIdx.x; j < topK; j += blockDim.x) {
        float data = 0;

        for (int m = 0; m < dim; m++) {
            float tmp = residual1[i][j][m] - residual2[i][j][m];
            data += tmp * tmp;
        }
        if (debug_flag & PRINT_DISTANCE) {
            printf("l2 dis %f\t", data);
            if (j == residual1.getSize(1) - 1) {
                printf("\n");
            }
        }
        outCodeDistances[i][j] = data;
    }
}

__device__ void printArray(
        Tensor<int, 2, true> codeDistances,
        const char* str) {
    printf("%s\n", str);
    for (int i = 0; i < codeDistances.getSize(0); ++i) {
        for (int j = 0; j < codeDistances.getSize(1); ++j) {
            printf("%d\t", (int)codeDistances[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void printArray(
        Tensor<float, 2, true> codeDistances,
        const char* str) {
    printf("%s\n", str);
    for (int i = 0; i < codeDistances.getSize(0); ++i) {
        for (int j = 0; j < codeDistances.getSize(1); ++j) {
            printf("%f\t", (float)codeDistances[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void sortByDistance(
        Tensor<float, 2, true> codeDistances,
        Tensor<Index::idx_t, 2, true> codeIndices) {
    int nq = codeDistances.getSize(0);
    int topK = codeDistances.getSize(1);
    int i = blockIdx.x;
    for (int j = 1; j < topK; j++) {
        float key = codeDistances[i][j];
        Index::idx_t value = codeIndices[i][j];
        int k = j - 1;
        while (k >= 0 && codeDistances[i][k] > key) {
            codeDistances[i][k + 1] = (float)codeDistances[i][k];
            codeIndices[i][k + 1] = (int)codeIndices[i][k];
            k--;
        }
        codeDistances[i][k + 1] = key;
        codeIndices[i][k + 1] = value;
    }

    // printArray(codeDistances, "codeDistances");
}

template <typename T>
__host__ void extractData(
        Tensor<T, 2, true>& from,
        Tensor<T, 2, true>& to,
        cudaStream_t stream) {
    int n1 = to.getSize(0);
    int n2 = to.getSize(1);
    for (int i = 0; i < n1; i++) {
        //        for (int j = 0; j < n2; j++) {
        //            T tmp = from[i][j];
        //            to[i][j] = tmp;
        //        }
        //        cudaMemcpyAsync(
        //                from[i].data(),
        //                to[i].data(),
        //                n2 * sizeof(T),
        //                cudaMemcpyDeviceToDevice,
        //                stream);
        fromDevice(from[i].data(), to[i].data(), n2, stream);
    }
}

template <typename T>
__global__ void extractData2(
        Tensor<T, 2, true> from,
        Tensor<T, 2, true> to,
        cudaStream_t stream) {
    int n1 = to.getSize(0);
    int n2 = to.getSize(1);
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < n2; j += blockDim.x) {
        T tmp = from[i][j];
        to[i][j] = tmp;
    }
    //        cudaMemcpyAsync(
    //                from[i].data(),
    //                to[i].data(),
    //                n2 * sizeof(T),
    //                cudaMemcpyDeviceToDevice,
    //                stream);
    // fromDevice(from[i].data(), to[i].data(), n2 * sizeof(T), stream);
}


template <typename T>
__global__ void extractIndex(
        Tensor<T, 2, true> from,
        Tensor<int, 2, true> fromIndex,
        Tensor<T, 2, true> to,
        cudaStream_t stream) {
    int n1 = to.getSize(0);
    int n2 = to.getSize(1);
    int i = blockIdx.x;
    for (int j = threadIdx.x; j < n2; j += blockDim.x) {
        T tmp = from[i][fromIndex[i][j]];
        to[i][j] = tmp;
    }
}

void IVFPQR::setPrecomputedCodes(bool enable) {
    if (enable) {
        IVFPQ::setPrecomputedCodes(enable); // todo:
    } else {
        IVFPQ::setPrecomputedCodes(enable);
    }
}

void IVFPQR::query(
        Tensor<float, 2, true>& queries,
        int nprobe,
        int topK,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices) {
    // indicesOptions_ = INDICES_IVF;
    if (debug_flag & PRINT_TIME) {
        cout << "use ivfpqr::query" << endl;
    }
    StopWatch sw = StopWatch::start();
    int realK = kFactor * topK;
    // These are caught at a higher level
    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(realK <= GPU_MAX_SELECTION_K);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    int nq = queries.getSize(0);

    DeviceTensor<float, 2, true> tmp_distances(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, realK});
    DeviceTensor<Index::idx_t, 2, true> tmp_labels(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, realK});
    Tensor<float, 2, true> tmpOutDistances(tmp_distances.data(), {nq, realK});
    Tensor<Index::idx_t, 2, true> tmpOutIndices(
            const_cast<Index::idx_t*>(tmp_labels.data()), {nq, realK});

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
            makeDevAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    DeviceTensor<int, 2, true> coarseIndices(
            resources_,
            makeDevAlloc(AllocType::Other, stream),
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
        if (debug_flag & PRINT_TIME) {
            printf("use precomputedCodes_\n");
        }
        FAISS_ASSERT(metric_ == MetricType::METRIC_L2);

        runPQPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                realK,
                tmpOutDistances,
                tmpOutIndices);
    } else {
        runPQNoPrecomputedCodes_(
                queries,
                coarseDistances,
                coarseIndices,
                realK,
                tmpOutDistances,
                tmpOutIndices);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "ivfpq::query done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    //残差
    DeviceTensor<float, 3, true> residual1(
            resources_,
            makeDevAlloc(AllocType::Other, stream),
            {nq, realK, dim_});

    DeviceTensor<int, 2, true> listIds(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, realK});
    DeviceTensor<int, 2, true> listOffsets(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, realK});
    {
        auto grid = dim3(nq);
        auto block = dim3(min(256, realK));
        calculateListId<<<grid, block, 0, stream>>>(
                listIds, listOffsets, tmpOutIndices, debug_flag);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "calculateListId done " << sw.getElapsedTime() << endl;
    }
    sw.restart();
    DeviceTensor<float, 3, true> listCoarseCentroids(
            resources_,
            makeDevAlloc(AllocType::Other, stream),
            {nq, realK, dim_});

    //计算query和"result所在的粗聚类"的残差
    quantizer_->reconstruct(listIds, listCoarseCentroids);
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "reconstruct done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    {
        auto grid = dim3(nq, realK);
        auto block = dim3(min(dim_, 256));
        runPQResidualVector1<<<grid, block, 0, stream>>>(
                residual1,
                queries,
                listIds,
                listOffsets,
                deviceListDataPointers_.data().get(),
                pqCentroidsInnermostCode_,
                pqCentroidsMiddleCode_,
                listCoarseCentroids,
                debug_flag);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "runPQResidualVector1 done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    // If the GPU isn't storing indices (they are on the CPU side), we
    // need to perform the re-mapping here
    // FIXME: we might ultimately be calling this function with inputs
    // from the CPU, these are unnecessary copies
    if (indicesOptions_ == INDICES_CPU) {
        HostTensor<Index::idx_t, 2, true> hostOutIndices(tmpOutIndices, stream);

        ivfOffsetToUserIndex(
                hostOutIndices.data(),
                numLists_,
                hostOutIndices.getSize(0),
                hostOutIndices.getSize(1),
                listOffsetToUserIndex_);

        // Copy back to GPU, since the input to this function is on the
        // GPU
        tmpOutIndices.copyFrom(hostOutIndices, stream);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "convert to cpu index done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    DeviceTensor<float, 3, true> residual2(
            resources_,
            makeDevAlloc(AllocType::Other, stream),
            {nq, realK, dim_});
    refinePQ.calculateResidualVector2(tmpOutIndices, residual2);
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "calculateResidualVector2 done " << sw.getElapsedTime() << endl;
        sw.restart();
    }

    DeviceTensor<float, 2, true> codeDistances(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, realK});
    {
        auto grid = dim3(nq);
        auto block = dim3(min(256, realK));
        pqCodeDistances<<<grid, block, 0, stream>>>(
                residual1, residual2, codeDistances, debug_flag);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "pqCodeDistances done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    // printArray<<<grid, block, 0, stream>>>(codeDistances);


    DeviceTensor<int, 2, true> reRankIndices(
            resources_, makeDevAlloc(AllocType::Other, stream), {nq, topK});
    {
        runBlockSelect(
                codeDistances,
                outDistances,
                reRankIndices,
                false,
                topK,
                stream);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "sortByDistance done " << sw.getElapsedTime() << endl;
        sw.restart();
    }
    {
        auto grid = dim3(nq);
        auto block = dim3(min(256, topK));
        extractIndex<<<grid, block, 0, stream>>>(
                tmpOutIndices, reRankIndices, outIndices, stream);
    }
    if (debug_flag & PRINT_TIME) {
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        sw.stop();
        cout << "extractData done " << sw.getElapsedTime() << endl;
        sw.restart();
    }

    // outIndices.copyFrom(tmp_labels.transpose(0, 1)[0], stream);
}

} // namespace gpu
} // namespace faiss

