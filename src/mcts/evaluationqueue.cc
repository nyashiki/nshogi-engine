//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evaluationqueue.h"
#include "../globalconfig.h"

namespace nshogi {
namespace engine {
namespace mcts {

EvaluationQueue::EvaluationQueue(std::size_t MaxSize)
    : MaxQueueSize(MaxSize)
    , IsOpen(false) {
}

bool EvaluationQueue::isOpen() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return IsOpen;
}

void EvaluationQueue::open() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsOpen = true;
}

void EvaluationQueue::close() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsOpen = false;
    }
    CV.notify_all();
}

std::size_t EvaluationQueue::count() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Queue.size();
}

bool EvaluationQueue::add(const core::State& State,
                                    const core::StateConfig& Config, Node* N) {
    global_config::FeatureType FSC(State, Config);

    std::unique_lock<std::mutex> Lock(Mutex);

    CV.wait(Lock, [this]() { return Queue.size() < MaxQueueSize || !IsOpen; });

    if (IsOpen) {
        Queue.emplace(State.getSideToMove(), N, std::move(FSC),
                      State.getHash());
        return true;
    }

    return false;
}

auto EvaluationQueue::get(std::size_t NumElements)
    -> std::tuple<std::vector<core::Color>, std::vector<Node*>,
                  std::vector<global_config::FeatureType>, std::vector<uint64_t>> {
    std::tuple<std::vector<core::Color>, std::vector<Node*>,
               std::vector<global_config::FeatureType>, std::vector<uint64_t>>
        T;

    std::get<0>(T).reserve(NumElements);
    std::get<1>(T).reserve(NumElements);
    std::get<2>(T).reserve(NumElements);
    std::get<3>(T).reserve(NumElements);

    {
        std::lock_guard<std::mutex> Lock(Mutex);
        while (!Queue.empty() && NumElements--) {
            auto Element = std::move(Queue.front());
            Queue.pop();

            std::get<0>(T).emplace_back(std::get<0>(Element));
            std::get<1>(T).emplace_back(std::get<1>(Element));
            std::get<2>(T).emplace_back(std::move(std::get<2>(Element)));
            std::get<3>(T).emplace_back(std::get<3>(Element));
        }
    }

    CV.notify_all();
    return T;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
