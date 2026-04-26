//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "extractbit.h"
#include <assert.h>

namespace {

__global__ void extractBitsKernelChannelsFirst(int* Dest, const uint64_t* Src, int BatchSize, int NumChannels) {
    int BitIndex = threadIdx.x;
    int Index = blockIdx.x;

    if (Index < BatchSize * NumChannels && BitIndex < 81) {
        int Rotate = (Src[2 * Index + 1] >> 24) & 1;
        int Value = Src[2 * Index + 1] >> 32;

        // Determine the target square based on the rotation value.
        // If Rotate is 0, the target is BitIndex; if Rotate is 1, it's 80 - BitIndex.
        // Using arithmetic to avoid conditionals for performance reasons.
        const int TargetSquare = BitIndex * (1 - 2 * Rotate) + 80 * Rotate;

        // Determine the shift amount needed to target the correct bit.
        // Using arithmetic to adjust for bit positions beyond 63.
        const int ShiftAmount = TargetSquare - 63 * (TargetSquare >= 63);

        // Identify the 64-bit chunk containing the target bit.
        // Using pointer arithmetic to avoid conditionals for performance reasons.
        const uint64_t* Target64BitPtr = &Src[2 * Index] + (TargetSquare >= 63);

        const uint64_t Mask = 1ULL << ShiftAmount;
        Dest[Index * 81 + BitIndex] = ((*Target64BitPtr & Mask) >> ShiftAmount) * Value;
    }
}

__global__ void extractBitsKernelChannelsLast(int* __restrict__ Dest, const uint64_t* __restrict__ Src, int BatchSize, int NumChannels) {
    const int ChannelIndex = threadIdx.x;
    const int BitIndex = blockIdx.x;
    const int BatchIndex = blockIdx.y;

    if (ChannelIndex < NumChannels && BitIndex < 81 && BatchIndex < BatchSize) {
        assert(((uintptr_t)Src & 0xF) == 0); // Ensure Src is 16-byte aligned.
        const ulonglong2 Source = reinterpret_cast<const ulonglong2*>(Src)[BatchIndex * NumChannels + ChannelIndex];
        const uint64_t Low = Source.x;
        const uint64_t High = Source.y;

        const int Rotate = (int)((High >> 24) & 1ULL);
        const int Value = (int)(High >> 32);

        const int TargetSquare = BitIndex * (1 - 2 * Rotate) + 80 * Rotate;

        const int UseHigh = (int)(TargetSquare >= 63);
        const int ShiftAmount = TargetSquare - 63 * UseHigh;

        // This ternary operation will be converted to `selp.u64` in PTX.
        // Thus, it avoids branching and is efficient on the GPU.
        const uint64_t Target = UseHigh ? High : Low;
        const uint64_t Mask = 1ULL << ShiftAmount;

        Dest[(BatchIndex * 81 + BitIndex) * NumChannels + ChannelIndex] =
            ((Target & Mask) >> ShiftAmount) * Value;
    }
}

} // namespace

namespace nshogi {
namespace engine {
namespace cuda {

template <bool ChannelsFirst>
void extractBits(float* Dest, const uint64_t* Src, int BatchSize, int NumChannels, cudaStream_t Stream) {
    assert(BatchSize * NumChannels > 0);

    // The pointer conversion from float* to int* is a technique used here.
    // The goal is to set the value in FeatureBitboard directly into the float array.
    // By treating the destination as an int*, we directly copy the memory layout from the
    // bits shifted by 32 from the 64-bit integer source.
    // If we were to directly use the 64-bit integer as a float, the value would become corrupt.
    // This ensures that the bit pattern of the integer is accurately represented in the float memory layout.
    if constexpr (ChannelsFirst) {
        dim3 Blocks(BatchSize * NumChannels, 1, 1);
        dim3 Threads(81, 1, 1);
        extractBitsKernelChannelsFirst<<<Blocks, Threads, 0, Stream>>>((int*)Dest, Src, BatchSize, NumChannels);
    } else {
        assert(NumChannels <= 1024);
        dim3 Blocks(81, BatchSize, 1);
        dim3 Threads(NumChannels, 1, 1);
        extractBitsKernelChannelsLast<<<Blocks, Threads, 0, Stream>>>((int*)Dest, Src, BatchSize, NumChannels);
    }
}

template
void extractBits<true>(float* Dest, const uint64_t* Src, int BatchSize, int NumChannels, cudaStream_t Stream);
template
void extractBits<false>(float* Dest, const uint64_t* Src, int BatchSize, int NumChannels, cudaStream_t Stream);

} // namespace cuda
} // namespace engine
} // namespace nshogi
