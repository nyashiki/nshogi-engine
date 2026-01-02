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
    Node(NodeIndex, float WinRate, float DrawRate);

    uint64_t visitCount() const;

    NodeIndex Index;

    std::vector<float> PolicyRaw;
    float WinRateRaw;
    float DrawRateRaw;

    std::vector<core::Move32> Moves;
    std::vector<uint64_t> VisitCounts;
    std::vector<double> WinRateAccumulateds;
    std::vector<double> DrawRateAccumulateds;
    std::vector<NodeIndex> Children;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // NSHOGI_ENGINE_BOOK_NODE_H
