//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evaluatequeue.h"

#include "../evaluate/preset.h"

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
EvaluationQueue<Features>::EvaluationQueue(std::size_t MaxSize)
    : MaxQueueSize(MaxSize)
    , IsOpen(false) {
}

template <typename Features>
void EvaluationQueue<Features>::open() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsOpen = true;
}

template <typename Features>
void EvaluationQueue<Features>::close() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsOpen = false;
    }
    CV.notify_all();
}

template <typename Features>
std::size_t EvaluationQueue<Features>::count() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Queue.size();
}

template <typename Features>
void EvaluationQueue<Features>::add(const core::State& State,
                                    const core::StateConfig& Config, Node* N) {
    Features FSC(State, Config);

    std::unique_lock<std::mutex> Lock(Mutex);

    CV.wait(Lock, [this]() { return Queue.size() < MaxQueueSize || !IsOpen; });

    if (IsOpen) {
        Queue.emplace(State.getSideToMove(), N, std::move(FSC),
                      State.getHash());
    }
}

template <typename Features>
auto EvaluationQueue<Features>::get(std::size_t NumElements)
    -> std::tuple<std::vector<core::Color>, std::vector<Node*>,
                  std::vector<Features>, std::vector<uint64_t>> {
    std::tuple<std::vector<core::Color>, std::vector<Node*>,
               std::vector<Features>, std::vector<uint64_t>>
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

template class EvaluationQueue<evaluate::preset::SimpleFeatures>;
template class EvaluationQueue<evaluate::preset::CustomFeaturesV1>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
