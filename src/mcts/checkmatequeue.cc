//
// Copyright (c) 2025-2026 @nyashiki
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

CheckmateQueue::CheckmateQueue()
    : QueueMaxSize(1UL * 1024UL * 1024 * 1024 / sizeof(CheckmateTask)) // 1 GB.
    , Generation(0) {
}

void CheckmateQueue::add(Node* N, const core::Position& Position,
                         uint64_t Depth) noexcept {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    if (Queue.size() < QueueMaxSize) {
        Queue.emplace(
            std::make_unique<CheckmateTask>(N, Position, Depth, Generation));
    }
}

bool CheckmateQueue::tryAdd(Node* N, const core::Position& Position,
                            uint64_t Depth) noexcept {
    bool Succeeded = SpinLock.tryLock();

    if (!Succeeded) {
        return false;
    }

    if (Queue.size() < QueueMaxSize) {
        Queue.emplace(
            std::make_unique<CheckmateTask>(N, Position, Depth, Generation));
    } else {
        Succeeded = false;
    }

    SpinLock.unlock();
    return Succeeded;
}

auto CheckmateQueue::get() noexcept -> std::unique_ptr<CheckmateTask> {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);

    if (Queue.empty()) {
        return nullptr;
    }

    std::unique_ptr<CheckmateTask> Task = std::move(Queue.front());
    Queue.pop();
    return Task;
}

auto CheckmateQueue::getAll() noexcept
    -> std::queue<std::unique_ptr<CheckmateTask>> {
    std::queue<std::unique_ptr<CheckmateTask>> Q;

    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    Queue.swap(Q);

    return Q;
}

void CheckmateQueue::incrementGeneration() {
    std::lock_guard<lock::SpinLock> Lock(SpinLock);
    ++Generation;

    // Clear the queue.
    std::queue<std::unique_ptr<CheckmateTask>> Empty;
    Queue.swap(Empty);
}

void CheckmateQueue::lock() noexcept {
    SpinLock.lock();
}

void CheckmateQueue::unlock() noexcept {
    SpinLock.unlock();
}

uint64_t CheckmateQueue::_generation() {
    return Generation;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
