//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MATH_SCORE_H
#define NSHOGI_ENGINE_MATH_SCORE_H

#include <algorithm>

namespace nshogi {
namespace engine {
namespace math {
namespace score {

// The network predicts P(win | not draw) and P(draw). Search accumulates
// E = P(win) + P(draw) / 2 instead, so averaging and changing perspective
// remain linear. The draw value is applied only when selecting a move.
inline double toExpectedScore(double ConditionalWinRate, double DrawRate) {
    return (1.0 - DrawRate) * ConditionalWinRate + 0.5 * DrawRate;
}

// Also works on accumulated values, before division by the visit count.
inline double withDrawValue(double ExpectedScore, double DrawRate,
                            double DrawValue) {
    return ExpectedScore + (DrawValue - 0.5) * DrawRate;
}

// Used when a network evaluation needs to fall back to a parent's search
// statistics. The conditional rate is undefined for a certain draw.
inline double toConditionalWinRate(double ExpectedScore, double DrawRate) {
    if (DrawRate >= 1.0) {
        return 0.5;
    }
    return std::clamp((ExpectedScore - 0.5 * DrawRate) / (1.0 - DrawRate), 0.0,
                      1.0);
}

} // namespace score
} // namespace math
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MATH_SCORE_H
