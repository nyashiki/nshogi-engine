//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_FEEDQUEUE_H_
#define NSHOGI_ENGINE_MCTS_FEEDQUEUE_H_

#include "node.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace mcts {

struct Batch {
 public:
    Batch(std::size_t BatchSize, std::unique_ptr<core::Color[]>&& Colors,
          std::unique_ptr<Node*[]>&& Nodes,
          std::unique_ptr<uint64_t[]>&& Hashes,
          std::unique_ptr<float[]>&& Policies,
          std::unique_ptr<float[]>&& WinRates,
          std::unique_ptr<float[]>&& DrawRates)
        : Size(BatchSize)
        , MyColors(std::move(Colors))
        , MyNodes(std::move(Nodes))
        , MyHashes(std::move(Hashes))
        , MyPolicies(std::move(Policies))
        , MyWinRates(std::move(WinRates))
        , MyDrawRates(std::move(DrawRates)) {
    }

    std::size_t size() const {
        return Size;
    }

    core::Color color(std::size_t Index) const {
        return MyColors[Index];
    }

    Node* node(std::size_t Index) const {
        return MyNodes[Index];
    }

    uint64_t hash(std::size_t Index) const {
        return MyHashes[Index];
    }

    float* policy(std::size_t Index) const {
        return MyPolicies.get() + 27 * core::NumSquares * Index;
    }

    float winRate(std::size_t Index) const {
        return MyWinRates[Index];
    }

    float drawRate(std::size_t Index) const {
        return MyDrawRates[Index];
    }

 private:
    const std::size_t Size;

    std::unique_ptr<core::Color[]> MyColors;
    std::unique_ptr<Node*[]> MyNodes;
    std::unique_ptr<uint64_t[]> MyHashes;
    std::unique_ptr<float[]> MyPolicies;
    std::unique_ptr<float[]> MyWinRates;
    std::unique_ptr<float[]> MyDrawRates;
};

class FeedQueue {
 public:
    FeedQueue();

    void notifyEvaluationStarts();
    void notifyEvaluationStops();

    void add(std::unique_ptr<Batch>&& B);
    std::unique_ptr<Batch> get();

 private:
    bool EvaluationStopped;

    std::condition_variable CV;
    std::mutex Mutex;
    std::queue<std::unique_ptr<Batch>> Queue;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_FEEDQUEUE_H_
