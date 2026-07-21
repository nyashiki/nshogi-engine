//
// Copyright (c) 2025-2026 @nyashiki
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
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>

namespace nshogi {
namespace engine {
namespace mcts {

struct Batch {
 public:
    Batch(std::size_t BatchSize, std::unique_ptr<Node*[]>&& Nodes,
          std::unique_ptr<uint64_t[]>&& Hashes,
          std::unique_ptr<uint32_t[]>&& PolicyOffsets,
          std::unique_ptr<float[]>&& LegalPolicies,
          std::unique_ptr<float[]>&& WinRates,
          std::unique_ptr<float[]>&& DrawRates)
        : Size(BatchSize)
        , MyNodes(std::move(Nodes))
        , MyHashes(std::move(Hashes))
        , MyPolicyOffsets(std::move(PolicyOffsets))
        , MyLegalPolicies(std::move(LegalPolicies))
        , MyWinRates(std::move(WinRates))
        , MyDrawRates(std::move(DrawRates)) {
    }

    std::size_t size() const {
        return Size;
    }

    Node* node(std::size_t Index) const {
        return MyNodes[Index];
    }

    uint64_t hash(std::size_t Index) const {
        return MyHashes[Index];
    }

    // The policy logits of the legal moves of the position, aligned
    // with the edge order of node(Index) (node(Index)->getNumChildren()
    // entries). The consumer may update the values in place (e.g.
    // apply softmax).
    float* legalPolicy(std::size_t Index) const {
        return MyLegalPolicies.get() + MyPolicyOffsets[Index];
    }

    float winRate(std::size_t Index) const {
        return MyWinRates[Index];
    }

    float drawRate(std::size_t Index) const {
        return MyDrawRates[Index];
    }

 private:
    const std::size_t Size;

    std::unique_ptr<Node*[]> MyNodes;
    std::unique_ptr<uint64_t[]> MyHashes;
    std::unique_ptr<uint32_t[]> MyPolicyOffsets;
    std::unique_ptr<float[]> MyLegalPolicies;
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
