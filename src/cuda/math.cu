//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "math.h"

namespace {

__global__ void sigmoidKernel(float* Dest, const float* Src, uint32_t N) {
    uint32_t Index = threadIdx.x + blockIdx.x * blockDim.x;

    if (Index < N) {
        Dest[Index] = 1.0f / (1.0f + __expf(-Src[Index]));
    }
}

} // namespace

namespace nshogi {
namespace engine {
namespace cuda {

void sigmoid(float* Dest, const float* Src, std::size_t N, cudaStream_t Stream) {
    uint32_t ThreadsPerBlock = 256;
    uint32_t BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    sigmoidKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, Stream>>>(Dest, Src, N);
}

} // namespace cuda
} // namespace engine
} // namespace nshogi
