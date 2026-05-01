//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_CHECKMATETASK_H
#define NSHOGI_ENGINE_MCTS_CHECKMATETASK_H

#include "node.h"

#include <cinttypes>
#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace mcts {

struct CheckmateTask {
 public:
    CheckmateTask(Node* N, const core::Position& Pos, uint64_t MaxDepth,
                  uint64_t Gen)
        : TargetNode(N)
        , Position(Pos)
        , Depth(MaxDepth)
        , Generation(Gen) {
    }

    Node* node() {
        return TargetNode;
    }

    core::Position& position() {
        return Position;
    }

    uint64_t depth() const {
        return Depth;
    }

    uint64_t generation() const {
        return Generation;
    }

 private:
    Node* TargetNode;
    core::Position Position;
    uint64_t Depth;
    uint64_t Generation;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATETASK_H
