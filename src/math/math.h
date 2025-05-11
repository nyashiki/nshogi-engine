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
template<std::same_as<double> T>
inline bool isnan_(T X) {
    const auto Bits = std::bit_cast<std::uint64_t>(X);
    constexpr uint64_t ExponentMask = 0x7FF0'0000'0000'0000ULL;
    constexpr uint64_t FractionMask = 0x000F'FFFF'FFFF'FFFFULL;
    return ((Bits & ExponentMask) == ExponentMask) && ((Bits & FractionMask) != 0);
}

template<std::same_as<float> T>
inline bool isnan_(T X) {
    const auto Bits = std::bit_cast<std::uint32_t>(X);
    constexpr uint32_t ExponentMask = 0x7F80'0000U;
    constexpr uint32_t FractionMask = 0x007F'FFFFU;
    return ((Bits & ExponentMask) == ExponentMask) && ((Bits & FractionMask) != 0);
}

inline float sigmoid(float X) {
    return 1.0f / (1.0f + std::exp(-X));
}

} // namespace math
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MATH_MATH_H
