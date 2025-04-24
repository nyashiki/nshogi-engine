//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MATH_MATH_H
#define NSHOGI_ENGINE_MATH_MATH_H

#include <cmath>
#include <cstdint>

namespace nshogi {
namespace engine {
namespace math {

// Since std::isnan is not available when -ffast-math is specified,
// we manually check for NaN using bitwise comparison based on the IEEE 754
// standard.
inline bool isnan_(double X) {
    union {
        double D;
        uint64_t I;
    } U;
    U.D = X;
    const uint64_t ExponentMask = 0x7FF0000000000000ULL;
    const uint64_t FractionMask = 0x000FFFFFFFFFFFFFULL;
    return ((U.I & ExponentMask) == ExponentMask) &&
           ((U.I & FractionMask) != 0);
}

inline float sigmoid(float X) {
    return 1.0f / (1.0f + std::exp(-X));
}

} // namespace math
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MATH_MATH_H
