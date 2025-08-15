//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "feedqueue.h"

namespace nshogi {
namespace engine {
namespace mcts {

FeedQueue::FeedQueue() {
}

void FeedQueue::notifyEvaluationStarts() {
    std::lock_guard<std::mutex> Lock(Mutex);
    EvaluationStopped = false;
}

void FeedQueue::notifyEvaluationStops() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        EvaluationStopped = true;
    }
    CV.notify_all();
}

void FeedQueue::add(std::unique_ptr<Batch>&& B) {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Queue.emplace(std::move(B));
    }
    CV.notify_one();
}

std::unique_ptr<Batch> FeedQueue::get() {
    std::unique_lock<std::mutex> Lock(Mutex);

    CV.wait(Lock, [&]() { return !Queue.empty() || EvaluationStopped; });

    if (Queue.empty()) {
        return nullptr;
    }

    auto Element = std::move(Queue.front());
    Queue.pop();

    return Element;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
