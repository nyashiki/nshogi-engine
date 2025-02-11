//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "extractbit.h"

namespace {

__global__ void extractBitsKernel(int* Dest, const uint64_t* Src, int N) {
    int BitIndex = threadIdx.x;
    int Index = blockIdx.x;

    if (Index < N && BitIndex < 81) {
        int Rotate = (Src[2 * Index + 1] >> 24) & 1;
        int Value = Src[2 * Index + 1] >> 32;

        // Determine the target square based on the rotation value.
        // If Rotate is 0, the target is BitIndex; if Rotate is 1, it's 80 - BitIndex.
        // Using arithmetic to avoid conditionals for performance reasons.
        int TargetSquare = BitIndex * (1 - 2 * Rotate) + 80 * Rotate;

        // Determine the shift amount needed to target the correct bit.
        // Using arithmetic to adjust for bit positions beyond 63.
        int ShiftAmount = TargetSquare - 63 * (TargetSquare >= 63);

        uint64_t Mask = 1ULL << ShiftAmount;

        // Identify the 64-bit chunk containing the target bit.
        // Using pointer arithmetic to avoid conditionals for performance reasons.
        const uint64_t* Target64BitPtr = &Src[2 * Index] + (TargetSquare >= 63);

        Dest[Index * 81 + BitIndex] = ((*Target64BitPtr & Mask) >> ShiftAmount) * Value;
    }
}

} // namespace

namespace nshogi {
namespace engine {
namespace cuda {


void extractBits(float* Dest, const uint64_t* Src, int N, cudaStream_t Stream) {
    dim3 Blocks(N, 1, 1);
    dim3 Threads(128, 1, 1);

    // The pointer conversion from float* to int* is a technique used here.
    // The goal is to set the value in FeatureBitboard directly into the float array.
    // By treating the destination as an int*, we directly copy the memory layout from the
    // bits shifted by 32 from the 64-bit integer source.
    // If we were to directly use the 64-bit integer as a float, the value would become corrupt.
    // This ensures that the bit pattern of the integer is accurately represented in the float memory layout.
    extractBitsKernel<<<Blocks, Threads, 0, Stream>>>((int*)Dest, Src, N);
}


} // namespace cuda
} // namespace engine
} // namespace nshogi
