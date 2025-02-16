//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "zero.h"

#include <cstring>
#include <nshogi/ml/common.h>

namespace nshogi {
namespace engine {
namespace infer {

Zero::Zero() {
}

Zero::~Zero() {
}

void Zero::computeNonBlocking(const ml::FeatureBitboard*, std::size_t BatchSize,
                              float* DstPolicy, float* DstWinRate,
                              float* DstDrawRate) {
    std::memset(DstPolicy, 0, BatchSize * ml::MoveIndexMax * sizeof(float));
    std::memset(DstWinRate, 0, BatchSize * sizeof(float));
    std::memset(DstDrawRate, 0, BatchSize * sizeof(float));
}

void Zero::computeBlocking(const ml::FeatureBitboard*, std::size_t BatchSize,
                           float* DstPolicy, float* DstWinRate,
                           float* DstDrawRate) {
    computeNonBlocking(nullptr, BatchSize, DstPolicy, DstWinRate, DstDrawRate);
    await();
}

void Zero::await() {
}

bool Zero::isComputing() {
    return false;
}

} // namespace infer
} // namespace engine
} // namespace nshogi
