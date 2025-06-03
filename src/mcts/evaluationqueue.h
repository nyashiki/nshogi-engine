//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_EVALUATIONQUEUE_H
#define NSHOGI_ENGINE_MCTS_EVALUATIONQUEUE_H

#include "node.h"

#include <condition_variable>
#include <mutex>
#include <queue>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class EvaluationQueue {
 public:
    EvaluationQueue(std::size_t MaxSize);

    bool isOpen() const;
    void open();
    void close();
    bool add(const core::State&, const core::StateConfig&, Node*);
    auto get(std::size_t NumElements)
        -> std::tuple<std::vector<core::Color>, std::vector<Node*>,
                      std::vector<Features>, std::vector<uint64_t>>;
    std::size_t count();

 private:
    const std::size_t MaxQueueSize;
    bool IsOpen;
    mutable std::mutex Mutex;
    std::condition_variable CV;
    // Tuple of (side to move, node address, feature vector, hash of the state).
    std::queue<std::tuple<core::Color, Node*, Features, uint64_t>> Queue;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALUATIONQUEUE_H
