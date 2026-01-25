//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "node.h"

namespace nshogi {
namespace engine {
namespace book {

Node::Node(NodeIndex I)
    : Index(I)
    , WinRateRaw(0.0f)
    , DrawRateRaw(0.0f) {
}

Node::Node(NodeIndex I, float WinRate, float DrawRate)
    : Index(I)
    , WinRateRaw(WinRate)
    , DrawRateRaw(DrawRate) {
}

uint64_t Node::visitCount() const {
    uint64_t Total = 0;
    for (const auto& VC : VisitCounts) {
        Total += VC;
    }
    return Total + 1;
}

} // namespace book
} // namespace engine
} // namespace nshogi
