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
#include "node.h"
#include "statistics.h"
#include "../context.h"
#include "../limit.h"
#include "../logger/logger.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

#include <functional>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

class SearchWorker : public worker::Worker {
 public:
    SearchWorker(allocator::Allocator* NodeAllocator,
                 allocator::Allocator* EdgeAllocator, EvaluationQueue*,
                 CheckmateQueue*, EvalCache*, Statistics* Stat);
    ~SearchWorker();

    void updateRoot(const core::State&, const core::StateConfig&, Node*);
    void setBannedMoves(const std::vector<core::Move32>& Moves);

 protected:
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    bool doTask() override;

    Node* collectOneLeaf();
    int16_t expandLeaf(Node*);

    void evaluateByRule(Node*);

    void immediateUpdateByWin(Node*);
    void immediateUpdateByLoss(Node*);
    void immediateUpdateByDraw(Node*, float DrawValue);
    void immediateUpdate(Node*);

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
    EvaluationQueue* EQueue;
    CheckmateQueue* CQueue;
    EvalCache* ECache;
    Statistics* PStat;

    EvalCache::EvalInfo CacheEvalInfo;

    std::vector<core::Move32> BannedMoves;
};

class SearchWorkerMaster : public SearchWorker {
 public:
    SearchWorkerMaster(
        const Context*,
        allocator::Allocator* NodeAllocator,
        allocator::Allocator* EdgeAllocator,
        EvaluationQueue*,
        CheckmateQueue*,
        EvalCache*,
        Statistics*,
        std::function<void()> SearchStopCallback,
        std::shared_ptr<logger::Logger>
    );
    ~SearchWorkerMaster() override;

    void setLimit(const engine::Limit& L);

    void start() override;
    bool doTask() override;
    void issueStop();

    void enableImmediateLog();
    void disableImmediateLog();

 private:
    logger::PVLog getPVLog() const;
    void dumpPVLog(uint64_t Elapsed) const;

    bool isRootSolved() const;
    bool checkNodeLimit() const;
    bool checkMemoryBudget() const;
    bool checkThinkingTimeBudget(uint64_t Elapsed) const;
    bool hasMadeUpMind(uint64_t Elapsed);

    bool checkSearchToStop(uint64_t Elapsed);

    const Context* PContext;
    const std::function<void()> Callback;
    std::shared_ptr<logger::Logger> Logger;
    bool ImmediateLogEnabled;

    engine::Limit Limit;

    std::chrono::time_point<std::chrono::steady_clock> SearchStartTime;
    uint64_t NumNodesAtStart;
    uint64_t LogOutputPrevious;

    bool Exiting;
    std::mutex Mutex;
    std::atomic<bool> CallbackCalled;
    bool ToCallCallback;
    std::thread StopCallThread;
    std::condition_variable StopCV;

    // Variables for checking if we make up the best move.
    uint64_t MadeUpCheckElapsedPrevious;
    const Edge* BestEdgePrevious;
    std::vector<double> VisitsPrevious;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
