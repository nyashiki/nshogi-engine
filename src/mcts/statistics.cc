//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "statistics.h"

#include <algorithm>

namespace nshogi {
namespace engine {
namespace mcts {

Statistics::Statistics() {
    reset();
}

void Statistics::reset() {
    // Search worker.
    NumNullLeaf.store(0, std::memory_order_relaxed);
    NumNonLeaf.store(0, std::memory_order_relaxed);
    NumRepetition.store(0, std::memory_order_relaxed);
    NumCheckmate.store(0, std::memory_order_relaxed);
    NumFailedToAllocateNode.store(0, std::memory_order_relaxed);
    NumFailedToAllocateEdge.store(0, std::memory_order_relaxed);
    NumConflictNodeAllocation.store(0, std::memory_order_relaxed);
    NumPolicyGreedyEdge.store(0, std::memory_order_relaxed);
    NumSpeculativeEdge.store(0, std::memory_order_relaxed);
    NumSpeculativeFailedEdge.store(0, std::memory_order_relaxed);
    NumTooManyVirtualLossEdge.store(0, std::memory_order_relaxed);
    NumFirstUnvisitedChildEdge.store(0, std::memory_order_relaxed);
    NumBeingExtractedChildren.store(0, std::memory_order_relaxed);
    NumUCBSelectionFailedEdge.store(0, std::memory_order_relaxed);
    NumNullUCBMaxEdge.store(0, std::memory_order_relaxed);
    NumCanDeclare.store(0, std::memory_order_relaxed);
    NumOverMaxPly.store(0, std::memory_order_relaxed);
    NumSucceededToAddEvaluationQueue.store(0, std::memory_order_relaxed);
    NumFailedToAddEvaluationQueue.store(0, std::memory_order_relaxed);
    NumCacheHit.store(0, std::memory_order_relaxed);

    // Evaluation worker.
    EvaluationCount.store(0, std::memory_order_relaxed);
    BatchSizeAccumulated.store(0, std::memory_order_relaxed);

    // Checkmate worker.
    NumSolverWorked.store(0, std::memory_order_relaxed);
    SolverElapsedMax.store(0, std::memory_order_relaxed);
    SolverElapsedAccumulated.store(0, std::memory_order_relaxed);
}

uint64_t Statistics::numNullLeaf() const {
    return NumNullLeaf.load(std::memory_order_relaxed);
}

uint64_t Statistics::numNonLeaf() const {
    return NumNonLeaf.load(std::memory_order_relaxed);
}

uint64_t Statistics::numRepetition() const {
    return NumRepetition.load(std::memory_order_relaxed);
}

uint64_t Statistics::numCheckmate() const {
    return NumCheckmate.load(std::memory_order_relaxed);
}

uint64_t Statistics::numFailedToAllocateNode() const {
    return NumFailedToAllocateNode.load(std::memory_order_relaxed);
}

uint64_t Statistics::numFailedToAllocateEdge() const {
    return NumFailedToAllocateEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numConflictNodeAllocation() const {
    return NumConflictNodeAllocation.load(std::memory_order_relaxed);
}

uint64_t Statistics::numPolicyGreedyEdge() const {
    return NumPolicyGreedyEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numSpeculativeEdge() const {
    return NumSpeculativeEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numSpeculativeFailedEdge() const {
    return NumSpeculativeFailedEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numTooManyVirtualLossEdge() const {
    return NumTooManyVirtualLossEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numFirstUnvisitedChildEdge() const {
    return NumFirstUnvisitedChildEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numBeingExtractedChildren() const {
    return NumBeingExtractedChildren.load(std::memory_order_relaxed);
}

uint64_t Statistics::numUCBSelectionFailedEdge() const {
    return NumUCBSelectionFailedEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numNullUCBMaxEdge() const {
    return NumNullUCBMaxEdge.load(std::memory_order_relaxed);
}

uint64_t Statistics::numCanDeclare() const {
    return NumCanDeclare.load(std::memory_order_relaxed);
}

uint64_t Statistics::numOverMaxPly() const {
    return NumOverMaxPly.load(std::memory_order_relaxed);
}

uint64_t Statistics::numSucceededToAddEvaluationQueue() const {
    return NumSucceededToAddEvaluationQueue.load(std::memory_order_relaxed);
}

uint64_t Statistics::numFailedToAddEvaluationQueue() const {
    return NumFailedToAddEvaluationQueue.load(std::memory_order_relaxed);
}

uint64_t Statistics::numCacheHit() const {
    return NumCacheHit.load(std::memory_order_relaxed);
}

uint64_t Statistics::evaluationCount() const {
    return EvaluationCount.load(std::memory_order_relaxed);
}

uint64_t Statistics::batchSizeAccumulated() const {
    return BatchSizeAccumulated.load(std::memory_order_relaxed);
}

uint64_t Statistics::numSolverWorked() const {
    return NumSolverWorked.load(std::memory_order_relaxed);
}

uint64_t Statistics::solverElapsedMax() const {
    return SolverElapsedMax.load(std::memory_order_relaxed);
}

uint64_t Statistics::solverElapsedAccumulated() const {
    return SolverElapsedAccumulated.load(std::memory_order_relaxed);
}

void Statistics::incrementNumNullLeaf() {
    NumNullLeaf.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumNonLeaf() {
    NumNonLeaf.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumRepetition() {
    NumRepetition.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumCheckmate() {
    NumCheckmate.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumFailedToAllocateNode() {
    NumFailedToAllocateNode.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumFailedToAllocateEdge() {
    NumFailedToAllocateEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumConflictNodeAllocation() {
    NumConflictNodeAllocation.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumPolicyGreedyEdge() {
    NumPolicyGreedyEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumSpeculativeEdge() {
    NumSpeculativeEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumSpeculativeFailedEdge() {
    NumSpeculativeFailedEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumTooManyVirtualLossEdge() {
    NumTooManyVirtualLossEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumFirstUnvisitedChildEdge() {
    NumFirstUnvisitedChildEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumBeingExtractedChildren() {
    NumBeingExtractedChildren.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumUCBSelectionFailedEdge() {
    NumUCBSelectionFailedEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumNullUCBMaxEdge() {
    NumNullUCBMaxEdge.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumCanDeclare() {
    NumCanDeclare.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumOverMaxPly() {
    NumOverMaxPly.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumSucceededToAddEvaluationQueue() {
    NumSucceededToAddEvaluationQueue.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumFailedToAddEvaluationQueue() {
    NumFailedToAddEvaluationQueue.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementNumCacheHit() {
    NumCacheHit.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::incrementEvaluationCount() {
    EvaluationCount.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::addBatchSizeAccumulated(uint64_t BatchSize) {
    BatchSizeAccumulated.fetch_add(BatchSize, std::memory_order_relaxed);
}

void Statistics::incrementNumSolverWorked() {
    NumSolverWorked.fetch_add(1, std::memory_order_relaxed);
}

void Statistics::updateSolverElapsed(uint64_t Elapsed) {
    //  - We intentionally accept some loss of `Elapsed` samples due to
    //    possible timing issues. For this statistics-only path, approximate
    //    values are good enough and relaxed ordering keeps the hot code fast.
    //  - For now we compute the maximum with `std::max` on the loaded value.
    //    Once C++26 (P0493R3) is widely available, replace this with
    //      SolverElapsedMax.fetch_max(Elapsed, std::memory_order_relaxed);
    //    to make the update fully atomic.
    // TODO: switch to fetch_max (C++26).
    SolverElapsedMax =
        std::max(SolverElapsedMax.load(std::memory_order_relaxed), Elapsed);

    SolverElapsedAccumulated.fetch_add(Elapsed, std::memory_order_relaxed);
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
