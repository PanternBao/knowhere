/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

//
// This kernel is split into a separate compilation unit to cut down
// on compile time
//

namespace faiss {
namespace gpu {

//// This is warp divergence central, but this is really a final step
//// and happening a small number of times
//inline __device__ int binarySearchForBucket(
//        int* prefixSumOffsets,
//        int size,
//        int val) {
//    int start = 0;
//    int end = size;
//
//    while (end - start > 0) {
//        int mid = start + (end - start) / 2;
//
//        int midVal = prefixSumOffsets[mid];
//
//        // Find the first bucket that we are <=
//        if (midVal <= val) {
//            start = mid + 1;
//        } else {
//            end = mid;
//        }
//    }
//
//    // We must find the bucket that it is in
//    assert(start != size);
//
//    return start;
//}

template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ, bool Dir>
__global__ void pass2SelectLists(
        Tensor<float, 2, true> heapDistances,
        Tensor<int, 2, true> heapIndices,
        void** listIndices,
        Tensor<int, 2, true> prefixSumOffsets,
        Tensor<int, 2, true> topQueryToCentroid,
        int k,
        IndicesOptions opt,
        Tensor<float, 2, true> outDistances,
        Tensor<Index::idx_t, 2, true> outIndices,
        Tensor<Index::idx_t, 2, true> outIndices2) {
    extern __shared__ float arrays[];
    float* smemK = (float*)arrays;
    int* smemV = (int*)&smemK[ThreadsPerBlock * NumWarpQ / kWarpSize];

    //__shared__ float smemK[kNumWarps * NumWarpQ];
    //__shared__ int smemV[kNumWarps * NumWarpQ];

    constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
    BlockSelect<
            float,
            int,
            Dir,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kInit, -1, smemK, smemV, k);

    auto queryId = blockIdx.x;
    int num = heapDistances.getSize(1);
    int limit = utils::roundDown(num, kWarpSize);

    int i = threadIdx.x;
    auto heapDistanceStart = heapDistances[queryId];

    // BlockSelect add cannot be used in a warp divergent circumstance; we
    // handle the remainder warp below
    for (; i < limit; i += blockDim.x) {
        heap.add(heapDistanceStart[i], i);
    }

    // Handle warp divergence separately
    if (i < num) {
        heap.addThreadQ(heapDistanceStart[i], i);
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outDistances[queryId][i] = smemK[i];

        // `v` is the index in `heapIndices`
        // We need to translate this into an original user index. The
        // reason why we don't maintain intermediate results in terms of
        // user indices is to substantially reduce temporary memory
        // requirements and global memory write traffic for the list
        // scanning.
        // This code is highly divergent, but it's probably ok, since this
        // is the very last step and it is happening a small number of
        // times (#queries x k).
        int v = smemV[i];
        Index::idx_t index = -1;
        Index::idx_t index2 = -1;

        if (v != -1) {
            // `offset` is the offset of the intermediate result, as
            // calculated by the original scan.
            int offset = heapIndices[queryId][v];

            // In order to determine the actual user index, we need to first
            // determine what list it was in.
            // We do this by binary search in the prefix sum list.
            int probe = binarySearchForBucket(
                    prefixSumOffsets[queryId].data(),
                    prefixSumOffsets.getSize(1),
                    offset);

            // This is then the probe for the query; we can find the actual
            // list ID from this
            int listId = topQueryToCentroid[queryId][probe];

            // Now, we need to know the offset within the list
            // We ensure that before the array (at offset -1), there is a 0
            // value
            int listStart = *(prefixSumOffsets[queryId][probe].data() - 1);
            int listOffset = offset - listStart;
            // This gives us our final index
            if (opt == INDICES_32_BIT) {
                index = (Index::idx_t)((int*)listIndices[listId])[listOffset];
            } else if (opt == INDICES_64_BIT) {
                index = ((Index::idx_t*)listIndices[listId])[listOffset];
            } else if (opt == INDICES_GPU_ALL) {
                index = ((Index::idx_t*)listIndices[listId])[listOffset];
                index2 =
                        ((Index::idx_t)listId << 32 | (Index::idx_t)listOffset);
            } else {
                index = ((Index::idx_t)listId << 32 | (Index::idx_t)listOffset);
            }
        }
        // don't change assign order.
        if (opt == INDICES_GPU_ALL) {
            outIndices2[queryId][i] = index2;
        }
        outIndices[queryId][i] = index;
    }
}

template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ, bool Dir>
__global__ void pass2SelectListsUseGlobalMemory(
        Tensor<float, 2, true> heapDistances,
        Tensor<int, 2, true> heapIndices,
        void** listIndices,
        Tensor<int, 2, true> prefixSumOffsets,
        Tensor<int, 2, true> topQueryToCentroid,
        int k,
        IndicesOptions opt,
        Tensor<float, 2, true> outDistances,
        Tensor<Index::idx_t, 2, true> outIndices,
        Tensor<Index::idx_t, 2, true> outIndices2) {
    __shared__ float* mallocArray;
    float* smemK = NULL;
    int* smemV = NULL;

    if (threadIdx.x == 0) {
        mallocArray = (float*)malloc(
                ThreadsPerBlock * NumWarpQ / kWarpSize * (4 + 4));
        if (mallocArray == NULL) {
            printf("illegal memory！！！\n");
            asm("trap;");
            return;
        }
    }
    __syncthreads();

    smemK = (float*)mallocArray;
    smemV = (int*)&smemK[ThreadsPerBlock * NumWarpQ / kWarpSize];

    constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
    BlockSelect<
            float,
            int,
            Dir,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kInit, -1, smemK, smemV, k);

    auto queryId = blockIdx.x;
    int num = heapDistances.getSize(1);
    int limit = utils::roundDown(num, kWarpSize);

    int i = threadIdx.x;
    auto heapDistanceStart = heapDistances[queryId];

    // BlockSelect add cannot be used in a warp divergent circumstance; we
    // handle the remainder warp below
    for (; i < limit; i += blockDim.x) {
        heap.add(heapDistanceStart[i], i);
    }

    // Handle warp divergence separately
    if (i < num) {
        heap.addThreadQ(heapDistanceStart[i], i);
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outDistances[queryId][i] = smemK[i];

        // `v` is the index in `heapIndices`
        // We need to translate this into an original user index. The
        // reason why we don't maintain intermediate results in terms of
        // user indices is to substantially reduce temporary memory
        // requirements and global memory write traffic for the list
        // scanning.
        // This code is highly divergent, but it's probably ok, since this
        // is the very last step and it is happening a small number of
        // times (#queries x k).
        int v = smemV[i];
        Index::idx_t index = -1;
        Index::idx_t index2 = -1;

        if (v != -1) {
            // `offset` is the offset of the intermediate result, as
            // calculated by the original scan.
            int offset = heapIndices[queryId][v];

            // In order to determine the actual user index, we need to first
            // determine what list it was in.
            // We do this by binary search in the prefix sum list.
            int probe = binarySearchForBucket(
                    prefixSumOffsets[queryId].data(),
                    prefixSumOffsets.getSize(1),
                    offset);

            // This is then the probe for the query; we can find the actual
            // list ID from this
            int listId = topQueryToCentroid[queryId][probe];

            // Now, we need to know the offset within the list
            // We ensure that before the array (at offset -1), there is a 0
            // value
            int listStart = *(prefixSumOffsets[queryId][probe].data() - 1);
            int listOffset = offset - listStart;

            // This gives us our final index
            if (opt == INDICES_32_BIT) {
                index = (Index::idx_t)((int*)listIndices[listId])[listOffset];
            } else if (opt == INDICES_64_BIT) {
                index = ((Index::idx_t*)listIndices[listId])[listOffset];
            } else if (opt == INDICES_GPU_ALL) {
                index = ((Index::idx_t*)listIndices[listId])[listOffset];
                index2 =
                        ((Index::idx_t)listId << 32 | (Index::idx_t)listOffset);
            } else {
                index = ((Index::idx_t)listId << 32 | (Index::idx_t)listOffset);
            }
        }
        // don't change assign order.
        if (opt == INDICES_GPU_ALL) {
            outIndices2[queryId][i] = index2;
        }
        outIndices[queryId][i] = index;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        // printf("free blockId %d\n", blockId);
        free(mallocArray);
    }
}

void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<int, 2, true>& heapIndices,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<int, 2, true>& topQueryToCentroid,
        int k,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        cudaStream_t stream,
        Tensor<Index::idx_t, 2, true>& outIndices2) {
    auto grid = dim3(topQueryToCentroid.getSize(0));

#define RUN_PASS(BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR)                      \
    do {                                                                    \
        const int use_memory = NUM_WARP_Q * BLOCK / kWarpSize * (4 + 4);    \
                                                                            \
        if (use_memory > 48 * 1024 && use_memory <= 64 * 1024) {            \
            cudaFuncSetAttribute(                                           \
                    pass2SelectLists<BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize,            \
                    use_memory);                                            \
            printf("kWarpSize %d，NUM_WARP_Q %d\n", kWarpSize, NUM_WARP_Q); \
        }                                                                   \
        if (use_memory <= 64 * 1024) {                                      \
            pass2SelectLists<BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR>          \
                    <<<grid,                                                \
                       BLOCK,                                               \
                       NUM_WARP_Q * BLOCK / kWarpSize*(4 + 4),              \
                       stream>>>(                                           \
                            heapDistances,                                  \
                            heapIndices,                                    \
                            listIndices.data().get(),                       \
                            prefixSumOffsets,                               \
                            topQueryToCentroid,                             \
                            k,                                              \
                            indicesOptions,                                 \
                            outDistances,                                   \
                            outIndices,                                     \
                            outIndices2);                                   \
        } else {                                                            \
            pass2SelectListsUseGlobalMemory<                                \
                    BLOCK,                                                  \
                    NUM_WARP_Q,                                             \
                    NUM_THREAD_Q,                                           \
                    DIR><<<grid, BLOCK, 0, stream>>>(                       \
                    heapDistances,                                          \
                    heapIndices,                                            \
                    listIndices.data().get(),                               \
                    prefixSumOffsets,                                       \
                    topQueryToCentroid,                                     \
                    k,                                                      \
                    indicesOptions,                                         \
                    outDistances,                                           \
                    outIndices,                                             \
                    outIndices2);                                           \
        }                                                                   \
        CUDA_TEST_ERROR();                                                  \
        return; /* success */                                               \
    } while (0)

#if GPU_MAX_SELECTION_K >= 2048

    // block size 128 for k <= 1024, 64 for k = 2048
#define RUN_PASS_DIR(DIR)                \
    do {                                 \
        if (k == 1) {                    \
            RUN_PASS(128, 1, 1, DIR);    \
        } else if (k <= 32) {            \
            RUN_PASS(128, 32, 2, DIR);   \
        } else if (k <= 64) {            \
            RUN_PASS(128, 64, 3, DIR);   \
        } else if (k <= 128) {           \
            RUN_PASS(128, 128, 3, DIR);  \
        } else if (k <= 256) {           \
            RUN_PASS(128, 256, 4, DIR);  \
        } else if (k <= 512) {           \
            RUN_PASS(128, 512, 8, DIR);  \
        } else if (k <= 1024) {          \
            RUN_PASS(128, 1024, 8, DIR); \
        } else if (k <= 2048) {          \
            RUN_PASS(64, 2048, 8, DIR);  \
        } else if (k <= 4096) {          \
            RUN_PASS(32, 4096, 8, DIR);  \
        } else if (k <= 8192) {          \
            RUN_PASS(32, 8192, 8, DIR);  \
        } else if (k <= 16384) {         \
            RUN_PASS(32, 16384, 8, DIR); \
        }                                \
    } while (0)

#else

#define RUN_PASS_DIR(DIR)                \
    do {                                 \
        if (k == 1) {                    \
            RUN_PASS(128, 1, 1, DIR);    \
        } else if (k <= 32) {            \
            RUN_PASS(128, 32, 2, DIR);   \
        } else if (k <= 64) {            \
            RUN_PASS(128, 64, 3, DIR);   \
        } else if (k <= 128) {           \
            RUN_PASS(128, 128, 3, DIR);  \
        } else if (k <= 256) {           \
            RUN_PASS(128, 256, 4, DIR);  \
        } else if (k <= 512) {           \
            RUN_PASS(128, 512, 8, DIR);  \
        } else if (k <= 1024) {          \
            RUN_PASS(128, 1024, 8, DIR); \
        }                                \
    } while (0)

#endif // GPU_MAX_SELECTION_K

    if (chooseLargest) {
        RUN_PASS_DIR(true);
    } else {
        RUN_PASS_DIR(false);
    }

    // unimplemented / too many resources
    FAISS_ASSERT_FMT(false, "unimplemented k value (%d)", k);

#undef RUN_PASS_DIR
#undef RUN_PASS
}

} // namespace gpu
} // namespace faiss
