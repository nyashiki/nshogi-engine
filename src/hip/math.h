//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_HIP_MATH_H
#define NSHOGI_ENGINE_HIP_MATH_H

#include <stdint.h>

#include <hip/hip_runtime.h>

namespace nshogi {
namespace engine {
namespace hip {

void sigmoid(float* Dest, const float* Src, std::size_t N,
             hipStream_t Stream = 0);

} // namespace hip
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_HIP_EXTRACTBIT_H
