//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
#define NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H

#include "checkmatetask.h"
#include "node.h"
#include "garbagecollector.h"

#include <memory>
#include <mutex>
#include <queue>

#include "../lock/spinlock.h"
#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace mcts {

class CheckmateQueue {
 public:
    CheckmateQueue(GarbageCollector*);

    void add(Node*, const core::Position&, uint64_t Depth) noexcept;
    bool tryAdd(Node*, const core::Position&, uint64_t Depth) noexcept;
    auto get() noexcept -> std::unique_ptr<CheckmateTask>;
    auto getAll() noexcept -> std::queue<std::unique_ptr<CheckmateTask>>;

    void incrementGeneration();

    void lock() noexcept;
    void unlock() noexcept;
    uint64_t _generation();

 private:
    GarbageCollector* GC;

    const std::size_t QueueMaxSize;
    uint64_t Generation;

    lock::SpinLock SpinLock;
    std::queue<std::unique_ptr<CheckmateTask>> Queue;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
