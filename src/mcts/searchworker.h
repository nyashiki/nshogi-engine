//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
#define NSHOGI_ENGINE_MCTS_SEARCHWORKER_H

#include "../context.h"
#include "../limit.h"
#include "../lock/spinlock.h"
#include "../logger/logger.h"
#include "../worker/worker.h"
#include "edge.h"
#include "evalcache.h"
#include "evaluationqueue.h"
#include "node.h"
#include "statistics.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/solver/dfpn.h>

#include <chrono>
#include <functional>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

class SearchWorker : public worker::Worker {
 public:
    SearchWorker(bool CheckmateSearchEnabled,
                 allocator::Allocator* NodeAllocator,
                 allocator::Allocator* EdgeAllocator, EvaluationQueue*,
                 EvalCache*, Statistics* Stat);
    ~SearchWorker();

    void updateRoot(const core::State&, const core::StateConfig&, Node*);

 protected:
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    // A single failed descent can be a transient race with another
    // worker; a short streak of them occurs while harvesting a freshly
    // fed frontier; only a long streak means every reachable leaf is
    // pending evaluation. On short streaks the worker briefly pauses and
    // retries: the pause must stay in the microsecond range because the
    // retries themselves (through the transient virtual losses they
    // leave behind) are what steers descents to free leaves and keeps
    // the evaluation pipeline saturated, while anything longer (a yield
    // or a sleep) throttles the retries so much that the in-flight
    // evaluations collapse to a single batch and the pipeline stages
    // serialize. Only on long streaks does the worker sleep to release
    // the core; the sleep must stay well below the feed interval since
    // slots for new descents open up per fed node and a sleeping worker
    // cannot see them until it wakes up.
    static constexpr uint32_t NullLeafStreakToPause = 3;
    static constexpr uint32_t NullLeafStreakToSleep = 1024;
    static constexpr std::chrono::microseconds NullLeafRetryPause{2};
    static constexpr std::chrono::microseconds NullLeafSleep{50};

    bool doTask() override;

    Node* collectOneLeaf();
    int16_t expandLeaf(Node*);

    void evaluateByRule(Node*);

    void immediateUpdateByWin(Node*);
    void immediateUpdateByLoss(Node*);
    void immediateUpdateByDraw(Node*);
    void immediateUpdate(Node*);

    Edge* computeUCBMaxEdge(Node*, uint16_t NumChildren, uint64_t MyVirtualLoss,
                            bool regardNotVisitedWin);
    double computeWinRateOfChild(Node* Child, uint64_t ChildVisits,
                                 uint64_t ChildVirtualVisits) const;
    void incrementVirtualLosses(Node*);

    const bool MyCheckmateSearchEnabled;

    std::unique_ptr<core::State> State;
    core::StateConfig Config;
    Node* RootNode;
    uint16_t RootPly;
    core::Color RootSideToMove;

    allocator::Allocator* NA;
    allocator::Allocator* EA;
    EvaluationQueue* EQueue;
    EvalCache* ECache;
    solver::dfpn::Solver DfPnSolver;
    Statistics* PStat;
    uint32_t ConsecutiveNullLeaves;

    EvalCache::EvalInfo CacheEvalInfo;
};

class SearchWorkerMaster : public SearchWorker {
 public:
    SearchWorkerMaster(const Context*, bool CheckmateSearchEnabled,
                       allocator::Allocator* NodeAllocator,
                       allocator::Allocator* EdgeAllocator, EvaluationQueue*,
                       EvalCache*, Statistics*,
                       std::function<void()> SearchStopCallback,
                       std::shared_ptr<logger::Logger>);
    ~SearchWorkerMaster() override;

    // Call only before start(), after any previous search has been awaited.
    void setLimit(const engine::Limit& L);

    void start() override;
    // Wait for both the search loop and its asynchronous stop callback.
    void await() override;
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
    std::condition_variable CallbackDoneCV;

    // Variables for checking if we make up the best move.
    uint64_t MadeUpCheckElapsedPrevious;
    const Edge* BestEdgePrevious;
    std::vector<double> VisitsPrevious;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
