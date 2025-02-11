//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_LIMIT_H
#define NSHOGI_ENGINE_LIMIT_H

#include <cstdint>


namespace nshogi {
namespace engine {

struct Limit {
    uint32_t TimeLimitMilliSeconds = 0;
    uint32_t ByoyomiMilliSeconds = 0;
    uint32_t IncreaseMilliSeconds = 0;

    bool isNoLimit() const {
        return (TimeLimitMilliSeconds == 0) &&
            (ByoyomiMilliSeconds == 0) &&
            (IncreaseMilliSeconds == 0);
    }
};

constexpr Limit NoLimit {0, 0, 0};

} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_LIMIT_H
