//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "statistics.h"

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
    NumNullUCBMaxEdge.store(0, std::memory_order_relaxed);
    NumCanDeclare.store(0, std::memory_order_relaxed);
    NumOverMaxPly.store(0, std::memory_order_relaxed);
    NumSucceededToAddEvaluationQueue.store(0, std::memory_order_relaxed);
    NumFailedToAddEvaluationQueue.store(0, std::memory_order_relaxed);
    NumCacheHit.store(0, std::memory_order_relaxed);

    // Evaluation worker.
    EvaluationCount.store(0, std::memory_order_relaxed);
    BatchSizeAccumulated.store(0, std::memory_order_relaxed);
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

} // namespace mcts
} // namespace engine
} // namespace nshogi
