//
// Copyright (c) 2025 @nyashiki
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
#define NSHOGI_ENGINE_MCTS_MANAGER_H

#include "../allocator/fixed_allocator.h"
#include "../allocator/segregated_free_list.h"
#include "../context.h"
#include "../evaluate/preset.h"
#include "../globalconfig.h"
#include "../limit.h"
#include "checkmatequeue.h"
#include "checkmateworker.h"
#include "evalcache.h"
#include "evaluationqueue.h"
#include "evaluationworker.h"
#include "garbagecollector.h"
#include "searchworker.h"
#include "tree.h"
#include "watchdog.h"

#include <atomic>
#include <condition_variable>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

struct ThoughtLog {
    std::vector<std::pair<core::Move16, uint64_t>> VisitCounts;
};

class Manager {
 public:
    Manager(const Context*, std::shared_ptr<logger::Logger> Logger);
    ~Manager();

    void thinkNextMove(const core::State&, const core::StateConfig&,
                       engine::Limit,
                       std::function<void(core::Move32, std::unique_ptr<ThoughtLog>)> Callback);
    void interrupt();

 private:
    void setupAllocator();
    void setupGarbageCollector();
    void setupMutexPool();
    void setupSearchTree();
    void setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs,
                              std::size_t NumEvaluationWorkersPerGPU);
    void setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs,
                                std::size_t NumEvaluationWorkersPerGPU);
    void setupSearchWorkers(std::size_t NumSearchWorkers);
    void setupCheckmateQueue(std::size_t NumCheckmateWorkers);
    void setupCheckmateWorkers(std::size_t NumCheckmateWorkers);
    void setupEvalCache(std::size_t EvalCacheMB);
    void setupSupervisor();
    void setupWatchDog();

    void doSupervisorWork(bool CallCackback);
    void doWatchdogWork();

    void stopWorkers();
    core::Move32 getBestmove(Node* Root);
    bool checkMemoryBudgetForPondering();

    void watchdogStopCallback();

    const Context* PContext;
    allocator::FixedAllocator<sizeof(Node)> NodeAllocator;
    allocator::SegregatedFreeListAllocator EdgeAllocator;

    std::unique_ptr<Tree> SearchTree;
    std::unique_ptr<GarbageCollector> GC;
    std::unique_ptr<EvaluationQueue<global_config::FeatureType>> EQueue;
    std::unique_ptr<CheckmateQueue> CQueue;
    std::unique_ptr<EvalCache> ECache;
    std::unique_ptr<MutexPool<lock::SpinLock>> MtxPool;
    std::vector<std::unique_ptr<SearchWorker<global_config::FeatureType>>>
        SearchWorkers;
    std::vector<std::unique_ptr<EvaluationWorker<global_config::FeatureType>>>
        EvaluationWorkers;
    std::vector<std::unique_ptr<CheckmateWorker>> CheckmateWorkers;

    std::shared_ptr<logger::Logger> PLogger;

    bool WakeUpSupervisor;
    std::unique_ptr<std::thread> Supervisor;
    std::mutex MutexSupervisor;
    std::condition_variable CVSupervisor;
    std::unique_ptr<Watchdog> WatchdogWorker;
    std::atomic<bool> HasInterruptReceived;

    std::unique_ptr<core::State> CurrentState;
    std::unique_ptr<core::StateConfig> StateConfig;
    std::unique_ptr<engine::Limit> Limit;
    std::function<void(core::Move32, std::unique_ptr<ThoughtLog>)> BestMoveCallback;

    bool IsExiting;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
