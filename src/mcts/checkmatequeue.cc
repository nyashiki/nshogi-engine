//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "checkmatequeue.h"

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateQueue::CheckmateQueue(std::size_t MaxSize,
                               std::size_t NumCheckmateWorkers)
    : QueueMaxSize(MaxSize)
    , IsOpen(false) {
}

void CheckmateQueue::open() {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    IsOpen = true;
}

void CheckmateQueue::close() {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    IsOpen = false;
}

void CheckmateQueue::add(Node* N, const core::Position& Position,
                         uint64_t Depth) noexcept {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    if (IsOpen) {
        if (Queue.size() < QueueMaxSize) {
            Queue.emplace(
                std::make_unique<CheckmateTask>(N, Position, Depth));
        }
    }
}

bool CheckmateQueue::tryAdd(Node* N, const core::Position& Position,
                            uint64_t Depth) noexcept {
    bool Succeeded = SpinLock.tryLock();

    if (!Succeeded) {
        return false;
    }

    if (IsOpen) {
        if (Queue.size() < QueueMaxSize) {
            Queue.emplace(
                std::make_unique<CheckmateTask>(N, Position, Depth));
        } else {
            Succeeded = false;
        }
    }

    SpinLock.unlock();
    return Succeeded;
}

auto CheckmateQueue::get(std::size_t WorkerId) noexcept
    -> std::unique_ptr<CheckmateTask> {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);

    if (Queue.empty()) {
        return nullptr;
    }

    std::unique_ptr<CheckmateTask> Task = std::move(Queue.front());
    Queue.pop();
    return Task;
}

auto CheckmateQueue::getAll(std::size_t WorkerId) noexcept
    -> std::queue<std::unique_ptr<CheckmateTask>> {
    std::queue<std::unique_ptr<CheckmateTask>> Q;

    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    Queue.swap(Q);

    return Q;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
