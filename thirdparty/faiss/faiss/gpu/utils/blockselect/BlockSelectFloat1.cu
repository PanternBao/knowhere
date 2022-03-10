/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

BLOCK_SELECT_IMPL(float, true, 1, 1);
BLOCK_SELECT_IMPL(float, false, 1, 1);

} // namespace gpu
} // namespace faiss