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

#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace mcts {

struct CheckmateTask {
 public:
    CheckmateTask(Node* N, const core::Position& Pos, uint64_t MaxDepth)
        : TargetNode(N)
        , Position(Pos)
        , Depth(MaxDepth) {
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

 private:
    Node* TargetNode;
    core::Position Position;
    uint64_t Depth;
};

class CheckmateQueue {
 public:
    CheckmateQueue(std::size_t MaxSize, std::size_t NumCheckmateWorkers);

    void open();
    void close();
    void add(Node*, const core::Position&, uint64_t Depth) noexcept;
    bool tryAdd(Node*, const core::Position&, uint64_t Depth) noexcept;
    auto getAll(std::size_t WorkerId) noexcept -> std::queue<std::unique_ptr<CheckmateTask>>;

 private:
    const std::size_t QueueMaxSize;
    const std::size_t NumWorkers;
    std::size_t RoundRobin;
    bool IsOpen;
    std::mutex GlobalMutex;
    std::vector<std::mutex> Mutexes;
    std::vector<std::queue<std::unique_ptr<CheckmateTask>>> Queues;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
