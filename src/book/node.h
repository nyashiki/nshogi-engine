//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_NODE_H
#define NSHOGI_ENGINE_BOOK_NODE_H

#include <vector>
#include <cstddef>
#include <cinttypes>

#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace book {

enum NodeIndex : std::size_t {
    NI_Null = 0,
    NI_Root = 1,
};

struct Node {
 public:
    Node(NodeIndex);
    Node(NodeIndex, NodeIndex Parent);
    Node(NodeIndex, NodeIndex Parent, float WinRate, float DrawRate);

    NodeIndex Index;

    uint64_t VisitCount;
    double WinRateAccumulated;
    double DrawRateAccumulated;
    float WinRateRaw;
    float DrawRateRaw;
    std::vector<NodeIndex> Parents;

    std::vector<core::Move32> Moves;
    std::vector<float> Policies;
    std::vector<NodeIndex> Children;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // NSHOGI_ENGINE_BOOK_NODE_H
