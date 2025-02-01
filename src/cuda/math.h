//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_CUDA_MATH_H
#define NSHOGI_ENGINE_CUDA_MATH_H


#include <stdint.h>

#include "cuda_runtime.h"

namespace nshogi {
namespace engine {
namespace cuda {

void sigmoid(float* Dest, const float* Src, std::size_t N, cudaStream_t Stream = 0);

} // namespace cuda
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_CUDA_EXTRACTBIT_H
