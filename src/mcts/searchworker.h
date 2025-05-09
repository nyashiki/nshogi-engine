//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
#define NSHOGI_ENGINE_MCTS_SEARCHWORKER_H

#include "../lock/spinlock.h"
#include "../worker/worker.h"
#include "checkmatequeue.h"
#include "edge.h"
#include "evalcache.h"
#include "evaluationqueue.h"
#include "mutexpool.h"
#include "node.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class SearchWorker : public worker::Worker {
 public:
    SearchWorker(allocator::Allocator* NodeAllocator,
                 allocator::Allocator* EdgeAllocator,
                 EvaluationQueue<Features>*, CheckmateQueue*,
                 MutexPool<lock::SpinLock>*, EvalCache*);
    ~SearchWorker();

    void updateRoot(const core::State&, const core::StateConfig&, Node*);
    Node* collectOneLeaf();
    int16_t expandLeaf(Node*);

    void evaluateByRule(Node*);

    void immediateUpdateByWin(Node*);
    void immediateUpdateByLoss(Node*);
    void immediateUpdateByDraw(Node*, float DrawValue);
    void immediateUpdate(Node*);

 private:
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    bool doTask() override;

    Edge* computeUCBMaxEdge(Node*, uint16_t NumChildren,
                            bool regardNotVisitedWin);
    double computeWinRateOfChild(Node* Child, uint64_t ChildVisits,
                                 uint64_t ChildVirtualVisits);
    void incrementVirtualLosses(Node*);

    std::unique_ptr<core::State> State;
    core::StateConfig Config;
    Node* RootNode;
    uint16_t RootPly;

    allocator::Allocator* NA;
    allocator::Allocator* EA;
    EvaluationQueue<Features>* EQueue;
    CheckmateQueue* CQueue;
    MutexPool<lock::SpinLock>* MtxPool;
    EvalCache* ECache;

    EvalCache::EvalInfo CacheEvalInfo;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
