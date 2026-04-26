//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
#define NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H

#include "node.h"

#include <memory>
#include <mutex>
#include <queue>

#include "../lock/spinlock.h"
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

class CheckmateQueue {
 public:
    CheckmateQueue();

    void add(Node*, const core::Position&, uint64_t Depth) noexcept;
    bool tryAdd(Node*, const core::Position&, uint64_t Depth) noexcept;
    auto get() noexcept -> std::unique_ptr<CheckmateTask>;
    auto getAll() noexcept -> std::queue<std::unique_ptr<CheckmateTask>>;

    void incrementGeneration();

    void lock() noexcept;
    void unlock() noexcept;
    uint64_t _generation();

 private:
    const std::size_t QueueMaxSize;
    uint64_t Generation;

    lock::SpinLock SpinLock;
    std::queue<std::unique_ptr<CheckmateTask>> Queue;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
