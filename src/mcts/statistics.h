//
// Copyright (c) 2025-2026 @nyashiki
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
#include <cstddef>

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
    uint64_t numSpeculativeFailedEdge() const;
    uint64_t numTooManyVirtualLossEdge() const;
    uint64_t numBeingExtractedChildren() const;
    uint64_t numUCBSelectionFailedEdge() const;
    uint64_t numNullUCBMaxEdge() const;
    uint64_t numCanDeclare() const;
    uint64_t numOverMaxPly() const;
    uint64_t numFailedToAddEvaluationQueue() const;
    uint64_t numCacheHit() const;

    uint64_t evaluationCount() const;
    uint64_t batchSizeAccumulated() const;

    uint64_t numSolverWorked() const;
    uint64_t solverElapsedMax() const;
    uint64_t solverElapsedAccumulated() const;

    void incrementNumNullLeaf();
    void incrementNumNonLeaf();
    void incrementNumRepetition();
    void incrementNumCheckmate();
    void incrementNumFailedToAllocateNode();
    void incrementNumFailedToAllocateEdge();
    void incrementNumConflictNodeAllocation();
    void incrementNumSpeculativeFailedEdge();
    void incrementNumTooManyVirtualLossEdge();
    void incrementNumBeingExtractedChildren();
    void incrementNumUCBSelectionFailedEdge();
    void incrementNumNullUCBMaxEdge();
    void incrementNumCanDeclare();
    void incrementNumOverMaxPly();
    void incrementNumFailedToAddEvaluationQueue();
    void incrementNumCacheHit();

    void incrementEvaluationCount();
    void addBatchSizeAccumulated(uint64_t BatchSize);

    void incrementNumSolverWorked();
    void updateSolverElapsed(uint64_t Elapsed);

 private:
    // Align each counter to its own cache line to avoid false sharing:
    // these counters are incremented by all workers in the search hot path.
    static constexpr std::size_t CacheLineSize = 64;

    // Search worker.
    alignas(CacheLineSize) std::atomic<uint64_t> NumNullLeaf;
    alignas(CacheLineSize) std::atomic<uint64_t> NumNonLeaf;
    alignas(CacheLineSize) std::atomic<uint64_t> NumRepetition;
    alignas(CacheLineSize) std::atomic<uint64_t> NumCheckmate;
    alignas(CacheLineSize) std::atomic<uint64_t> NumFailedToAllocateNode;
    alignas(CacheLineSize) std::atomic<uint64_t> NumFailedToAllocateEdge;
    alignas(CacheLineSize) std::atomic<uint64_t> NumConflictNodeAllocation;
    alignas(CacheLineSize) std::atomic<uint64_t> NumSpeculativeFailedEdge;
    alignas(CacheLineSize) std::atomic<uint64_t> NumTooManyVirtualLossEdge;
    alignas(CacheLineSize) std::atomic<uint64_t> NumBeingExtractedChildren;
    alignas(CacheLineSize) std::atomic<uint64_t> NumUCBSelectionFailedEdge;
    alignas(CacheLineSize) std::atomic<uint64_t> NumNullUCBMaxEdge;
    alignas(CacheLineSize) std::atomic<uint64_t> NumCanDeclare;
    alignas(CacheLineSize) std::atomic<uint64_t> NumOverMaxPly;
    alignas(CacheLineSize) std::atomic<uint64_t> NumFailedToAddEvaluationQueue;
    alignas(CacheLineSize) std::atomic<uint64_t> NumCacheHit;

    // Evaluation worker.
    alignas(CacheLineSize) std::atomic<uint64_t> EvaluationCount;
    alignas(CacheLineSize) std::atomic<uint64_t> BatchSizeAccumulated;

    // Checkmate worker.
    alignas(CacheLineSize) std::atomic<uint64_t> NumSolverWorked;
    alignas(CacheLineSize) std::atomic<uint64_t> SolverElapsedMax;
    alignas(CacheLineSize) std::atomic<uint64_t> SolverElapsedAccumulated;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_STATISTICS_H
