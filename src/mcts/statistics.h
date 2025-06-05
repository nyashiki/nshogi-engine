//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_STATISTICS_H
#define NSHOGI_ENGINE_MCTS_STATISTICS_H

#include <atomic>
#include <cinttypes>

namespace nshogi {
namespace engine {
namespace mcts {

class Statistics {
 public:
    Statistics();
    void reset();

    uint64_t numNullLeaf() const;
    uint64_t numNonLeaf() const;
    uint64_t numRepetition() const;
    uint64_t numCheckmate() const;
    uint64_t numFailedToAllocateNode() const;
    uint64_t numFailedToAllocateEdge() const;
    uint64_t numConflictNodeAllocation() const;
    uint64_t numNullUCBMaxEdge() const;
    uint64_t numCanDeclare() const;
    uint64_t numOverMaxPly() const;
    uint64_t numSucceededToAddEvaluationQueue() const;
    uint64_t numFailedToAddEvaluationQueue() const;
    uint64_t numCacheHit() const;

    uint64_t evaluationCount() const;
    uint64_t batchSizeAccumulated() const;

    void incrementNumNullLeaf();
    void incrementNumNonLeaf();
    void incrementNumRepetition();
    void incrementNumCheckmate();
    void incrementNumFailedToAllocateNode();
    void incrementNumFailedToAllocateEdge();
    void incrementNumConflictNodeAllocation();
    void incrementNumNullUCBMaxEdge();
    void incrementNumCanDeclare();
    void incrementNumOverMaxPly();
    void incrementNumSucceededToAddEvaluationQueue();
    void incrementNumFailedToAddEvaluationQueue();
    void incrementNumCacheHit();

    void incrementEvaluationCount();
    void addBatchSizeAccumulated(uint64_t BatchSize);

 private:
    // Search worker.
    std::atomic<uint64_t> NumNullLeaf;
    std::atomic<uint64_t> NumNonLeaf;
    std::atomic<uint64_t> NumRepetition;
    std::atomic<uint64_t> NumCheckmate;
    std::atomic<uint64_t> NumFailedToAllocateNode;
    std::atomic<uint64_t> NumFailedToAllocateEdge;
    std::atomic<uint64_t> NumConflictNodeAllocation;
    std::atomic<uint64_t> NumNullUCBMaxEdge;
    std::atomic<uint64_t> NumCanDeclare;
    std::atomic<uint64_t> NumOverMaxPly;
    std::atomic<uint64_t> NumSucceededToAddEvaluationQueue;
    std::atomic<uint64_t> NumFailedToAddEvaluationQueue;
    std::atomic<uint64_t> NumCacheHit;

    // Evaluation worker.
    std::atomic<uint64_t> EvaluationCount;
    std::atomic<uint64_t> BatchSizeAccumulated;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_STATISTICS_H
