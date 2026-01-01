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
    , VisitCount(0)
    , WinRateAccumulated(0.0)
    , DrawRateAccumulated(0.0)
    , WinRateRaw(0.0f)
    , DrawRateRaw(0.0f) {
}

Node::Node(NodeIndex I, NodeIndex Parent)
    : Index(I)
    , VisitCount(0)
    , WinRateAccumulated(0.0)
    , DrawRateAccumulated(0.0)
    , WinRateRaw(0.0f)
    , DrawRateRaw(0.0f) {
    Parents.push_back(Parent);
}

Node::Node(NodeIndex I, NodeIndex Parent, float WinRate, float DrawRate)
    : Index(I)
    , VisitCount(0)
    , WinRateAccumulated(0.0)
    , DrawRateAccumulated(0.0)
    , WinRateRaw(WinRate)
    , DrawRateRaw(DrawRate) {
    Parents.push_back(Parent);
}

} // namespace book
} // namespace engine
} // namespace nshogi
