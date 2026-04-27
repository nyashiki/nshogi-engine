//
// Copyright (c) 2025-2026 @nyashiki
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
#define NSHOGI_ENGINE_MCTS_MANAGER_H

#include "../allocator/fixed_allocator.h"
#include "../allocator/segregated_free_list.h"
#include "../book/book.h"
#include "../context.h"
#include "../globalconfig.h"
#include "../limit.h"
#include "checkmatequeue.h"
#include "checkmateworker.h"
#include "evalcache.h"
#include "evaluationqueue.h"
#include "evaluationworker.h"
#include "feedqueue.h"
#include "feedworker.h"
#include "garbagecollector.h"
#include "searchworker.h"
#include "statistics.h"
#include "tree.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

enum class ManagerStatus {
    Idle,
    Thinking,
    Pondering,
    Stopping,
    Busy,
};

enum class BookStrategyType {
    Top,
    Random,
};

class Manager {
 public:
    Manager(const Context*, std::shared_ptr<logger::Logger> Logger);
    ~Manager();

    void thinkNextMove(const core::State&, const core::StateConfig&,
                       engine::Limit,
                       std::function<void(core::Move32)> Callback,
                       std::function<void(Tree*)> SearchTreeCallback = nullptr);
    void interrupt();

    void resetSearchTree();

 private:
    void interruptInternal(bool Internal);

    void setupAllocator();
    void setupGarbageCollector();
    void setupSearchTree();
    void setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs,
                              std::size_t NumEvaluationWorkersPerGPU);
    void setupFeedQueue();
    void setupFeedWorkers(std::size_t NumFeedWorkers);
    void setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs,
                                std::size_t NumEvaluationWorkersPerGPU);
    void setupSearchWorkers(std::size_t NumSearchWorkers);
    void setupCheckmateQueue(std::size_t NumCheckmateWorkers);
    void setupCheckmateWorkers(std::size_t NumCheckmateWorkers);
    void setupEvalCache(std::size_t EvalCacheMB);
    void setupSupervisor();
    void setupBook();

    void doSupervisorWork(bool CallCackback);

    void stopWorkers();
    void awaitWorkers();
    core::Move32 getBestmove(Node* Root);
    bool checkMemoryBudgetForPondering();

    void searchStopCallback();

    bool checkAllVirtualLossIsZero(Node* Root) const;

    const Context* PContext;
    allocator::FixedAllocator<sizeof(Node)> NodeAllocator;
    allocator::SegregatedFreeListAllocator<> EdgeAllocator;

    std::unique_ptr<Tree> SearchTree;
    std::unique_ptr<GarbageCollector> GC;
    std::unique_ptr<EvaluationQueue> EQueue;
    std::unique_ptr<CheckmateQueue> CQueue;
    std::unique_ptr<FeedQueue> FQueue;
    std::unique_ptr<EvalCache> ECache;
    std::vector<std::unique_ptr<SearchWorker>> SearchWorkers;
    SearchWorkerMaster* SWorkerMaster;
    std::vector<std::unique_ptr<EvaluationWorker>> EvaluationWorkers;
    std::vector<std::unique_ptr<FeedWorker>> FeedWorkers;
    std::vector<std::unique_ptr<CheckmateWorker>> CheckmateWorkers;

    std::unique_ptr<book::Book> PBook;

    std::shared_ptr<logger::Logger> PLogger;

    bool WakeUpSupervisor;
    std::unique_ptr<std::thread> Supervisor;
    std::mutex MutexSupervisor;
    std::condition_variable CVSupervisor;

    std::mutex MutexStatus;
    std::condition_variable CVStatus;
    ManagerStatus Status;

    std::unique_ptr<core::State> CurrentState;
    std::unique_ptr<core::StateConfig> StateConfig;
    std::function<void(core::Move32)> BestMoveCallback;
    std::function<void(Tree*)> STCallback;

    Statistics Stat;

    bool IsExiting;

    const BookStrategyType BookStrategy = BookStrategyType::Top;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
