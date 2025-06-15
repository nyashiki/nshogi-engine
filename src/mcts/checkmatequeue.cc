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

CheckmateQueue::CheckmateQueue(std::size_t MaxSize, std::size_t NumCheckmateWorkers)
    : QueueMaxSize(MaxSize)
    , NumWorkers(NumCheckmateWorkers)
    , RoundRobin(0)
    , IsOpen(false)
    , Mutexes(NumWorkers)
    , Queues(NumWorkers) {

}

void CheckmateQueue::open() {
    std::lock_guard<std::mutex> Lock(GlobalMutex);
    IsOpen = true;
}

void CheckmateQueue::close() {
    std::lock_guard<std::mutex> Lock(GlobalMutex);
    IsOpen = false;
}

void CheckmateQueue::add(Node* N, const core::Position& Position, uint64_t Depth) noexcept {
    std::lock_guard<std::mutex> Lock(GlobalMutex);
    if (IsOpen) {
        std::lock_guard<std::mutex> LockWorker(Mutexes[RoundRobin]);
        if (Queues[RoundRobin].size() < QueueMaxSize) {
            Queues[RoundRobin].emplace(std::make_unique<CheckmateTask>(N, Position, Depth));
            RoundRobin = (RoundRobin + 1) % NumWorkers;
        }
    }
}

bool CheckmateQueue::tryAdd(Node* N, const core::Position& Position, uint64_t Depth) noexcept {
    bool Succeeded = GlobalMutex.try_lock();

    if (!Succeeded) {
        return false;
    }

    if (IsOpen) {
        std::lock_guard<std::mutex> LockWorker(Mutexes[RoundRobin]);
        if (Queues[RoundRobin].size() < QueueMaxSize) {
            Queues[RoundRobin].emplace(std::make_unique<CheckmateTask>(N, Position, Depth));
            RoundRobin = (RoundRobin + 1) % NumWorkers;
        } else {
            Succeeded = false;
        }
    }

    GlobalMutex.unlock();
    return Succeeded;
}

auto CheckmateQueue::getAll(std::size_t WorkerId) noexcept -> std::queue<std::unique_ptr<CheckmateTask>> {
    std::queue<std::unique_ptr<CheckmateTask>> Q;

    {
        std::lock_guard<std::mutex> Lock(Mutexes[WorkerId]);
        Queues[WorkerId].swap(Q);
    }

    return Q;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
