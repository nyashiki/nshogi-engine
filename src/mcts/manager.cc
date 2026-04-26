//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "manager.h"

#include <cmath>
#include <functional>

namespace nshogi {
namespace engine {
namespace mcts {

Manager::Manager(const Context* C, std::shared_ptr<logger::Logger> Logger)
    : PContext(C)
    , PLogger(std::move(Logger))
    , WakeUpSupervisor(false)
    , Status(ManagerStatus::Idle)
    , IsExiting(false) {
    std::thread AllocatorPrepareThread([&]() { setupAllocator(); });
    std::thread GarbageCollectorPrepareThread(
        [&]() { setupGarbageCollector(); });
    AllocatorPrepareThread.join();
    GarbageCollectorPrepareThread.join();

    setupSearchTree();
    setupCheckmateQueue(PContext->getNumCheckmateSearchThreads());
    setupFeedQueue();
    setupEvalCache(PContext->getEvalCacheMemoryMB());

    setupEvaluationQueue(PContext->getBatchSize(), PContext->getNumGPUs(),
                         PContext->getNumEvaluationThreadsPerGPU());
    setupEvaluationWorkers(PContext->getBatchSize(), PContext->getNumGPUs(),
                           PContext->getNumEvaluationThreadsPerGPU());
    setupFeedWorkers(PContext->getNumFeedThreads());
    setupCheckmateWorkers(PContext->getNumCheckmateSearchThreads());
    setupSearchWorkers(PContext->getNumSearchThreads());
    setupSupervisor();

    PLogger->setIsNShogiExtensionLogEnabled(
        PContext->isNShogiExtensionLogEnabled());
}

Manager::~Manager() {
    interruptInternal(true);
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->stop();
    }

    {
        std::lock_guard<std::mutex> LockS(MutexSupervisor);
        IsExiting = true;
    }
    CVSupervisor.notify_all();

    Supervisor->join();

    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }

    for (auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker.reset(nullptr);
    }
    for (auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker.reset(nullptr);
    }
    for (auto& SearchWorker : SearchWorkers) {
        SearchWorker.reset(nullptr);
    }
    for (auto& FeedWorker : FeedWorkers) {
        FeedWorker.reset(nullptr);
    }

    // SearchTree's destructor must be called before
    // GarbageCollector is released. Hence,
    // call the destructor explicitly here.
    SearchTree.reset(nullptr);
}

void Manager::thinkNextMove(const core::State& State,
                            const core::StateConfig& Config, engine::Limit Lim,
                            std::function<void(core::Move32)> Callback,
                            std::function<void(Tree*)> SearchTreeCallback) {

    interruptInternal(true);

    for (auto& CheckmateWorker : CheckmateWorkers) {
        if (!CheckmateWorker->isRunning()) {
            CheckmateWorker->start();
        }
    }

    {
        std::lock_guard<std::mutex> Lock(MutexStatus);
        Status = ManagerStatus::Busy;
    }

    // Update the current state.
    {
        CurrentState = std::make_unique<core::State>(State.clone());
        StateConfig = std::make_unique<core::StateConfig>(Config);
        BestMoveCallback = Callback;
        STCallback = SearchTreeCallback;
        std::lock_guard<std::mutex> Lock(MutexSupervisor);
        SWorkerMaster->setLimit(Lim);
        assert(!WakeUpSupervisor);
        WakeUpSupervisor = true;
        PLogger->setIsInverse(false);
    }

    // Wake up the supervisor.
    CVSupervisor.notify_one();
}

void Manager::interrupt() {
    interruptInternal(false);
}

void Manager::interruptInternal(bool Internal) {
    std::unique_lock<std::mutex> Lock(MutexStatus);
    CVStatus.wait(Lock, [&]() { return Status != ManagerStatus::Busy; });

    if (Status != ManagerStatus::Idle && Status != ManagerStatus::Stopping) {
        Status = ManagerStatus::Stopping;
        SWorkerMaster->issueStop();
        if (CQueue != nullptr) {
            CQueue->incrementGeneration();
        }

        if (!Internal) {
            for (auto& CheckmateWorker : CheckmateWorkers) {
                CheckmateWorker->stop();
                CheckmateWorker->await();
            }
        }
    }
}

void Manager::resetSearchTree() {
    SearchTree->reset();
}

void Manager::setupAllocator() {
    std::thread NodeAllocatorPrepareThread([&]() {
        NodeAllocator.resize(
            (std::size_t)(0.1 * (double)(PContext->getAvailableMemoryMB() *
                                         1024UL * 1024UL)));
    });

    std::thread EdgeAllocatorPrepareThread([&]() {
        EdgeAllocator.resize(
            (std::size_t)(0.9 * (double)(PContext->getAvailableMemoryMB() *
                                         1024UL * 1024UL)));
    });

    NodeAllocatorPrepareThread.join();
    EdgeAllocatorPrepareThread.join();
}

void Manager::setupGarbageCollector() {
    GC = std::make_unique<GarbageCollector>(
        PContext->getNumGarbageCollectorThreads(), &NodeAllocator,
        &EdgeAllocator);
}

void Manager::setupSearchTree() {
    SearchTree =
        std::make_unique<Tree>(GC.get(), &NodeAllocator, PLogger.get());
}

void Manager::setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs,
                                   std::size_t NumEvaluationWorkersPerGPU) {
    EQueue = std::make_unique<EvaluationQueue>(BatchSize * NumGPUs *
                                               NumEvaluationWorkersPerGPU * 64);
}

void Manager::setupFeedQueue() {
    FQueue = std::make_unique<FeedQueue>();
}

void Manager::setupFeedWorkers(std::size_t NumFeedWorkers) {
    for (std::size_t I = 0; I < NumFeedWorkers; ++I) {
        FeedWorkers.emplace_back(
            std::make_unique<FeedWorker>(PContext, FQueue.get(), ECache.get()));
    }
}

void Manager::setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs,
                                     std::size_t NumEvaluationWorkersPerGPU) {
    std::size_t ThreadId = 0;
    for (std::size_t I = 0; I < NumGPUs; ++I) {
        for (std::size_t J = 0; J < NumEvaluationWorkersPerGPU; ++J) {
            EvaluationWorkers.emplace_back(std::make_unique<EvaluationWorker>(
                PContext, ThreadId, I, BatchSize, EQueue.get(), FQueue.get(),
                &Stat));
            ++ThreadId;
        }
    }
}

void Manager::setupSearchWorkers(std::size_t NumSearchWorkers) {
    for (std::size_t I = 1; I < NumSearchWorkers; ++I) {
        SearchWorkers.emplace_back(std::make_unique<SearchWorker>(
            &NodeAllocator, &EdgeAllocator, EQueue.get(), CQueue.get(),
            ECache.get(), &Stat));
    }
    // IMPORTANT: add SearchWorkerMaster last in SearchWorkers because
    // the master controls when to stop all workers to search, stopWorkers(),
    // after its start(). stopWorkers() must be called after all
    // workers have started searching.
    // By adding the master as the last element
    // in `SearchWorkers`,
    //     for (auto& Worker : SearchWorkers) Worker->start()
    // naturally guarantees the master starts after the others start.
    SWorkerMaster = new SearchWorkerMaster(
        PContext, &NodeAllocator, &EdgeAllocator, EQueue.get(), CQueue.get(),
        ECache.get(), &Stat, std::bind(&Manager::searchStopCallback, this),
        PLogger);
    SearchWorkers.emplace_back(SWorkerMaster);
}

void Manager::setupCheckmateQueue(std::size_t NumCheckmateWorkers) {
    // CheckmateQueue must be initialized before SearchWorker is.
    assert(SearchWorkers.size() == 0);
    if (NumCheckmateWorkers > 0) {
        CQueue = std::make_unique<CheckmateQueue>();
    }
}

void Manager::setupCheckmateWorkers(std::size_t NumCheckmateWorkers) {
    assert(NumCheckmateWorkers == 0 || CQueue != nullptr);
    for (std::size_t I = 0; I < NumCheckmateWorkers; ++I) {
        CheckmateWorkers.emplace_back(
            std::make_unique<CheckmateWorker>(CQueue.get(), &Stat));
    }
}

void Manager::setupEvalCache(std::size_t EvalCacheMB) {
    if (EvalCacheMB > 0) {
        ECache = std::make_unique<EvalCache>(EvalCacheMB);
    }
}

void Manager::setupSupervisor() {
    Supervisor = std::make_unique<std::thread>([this]() {
        while (true) {
            {
                std::unique_lock<std::mutex> Lock(MutexSupervisor);

                CVSupervisor.wait(
                    Lock, [this]() { return WakeUpSupervisor || IsExiting; });

                if (IsExiting) {
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> Lock(MutexSupervisor);
                doSupervisorWork(true);
                WakeUpSupervisor = false;
            }
        }

        SWorkerMaster->issueStop();

        // Wait for all workers if previous search is running.
        awaitWorkers();
        assert(checkAllVirtualLossIsZero(SearchTree->getRoot()));

        {
            std::lock_guard<std::mutex> Lock(MutexStatus);
            Status = ManagerStatus::Idle;
        }
        CVStatus.notify_all();
    });
}

void Manager::doSupervisorWork(bool CallCallback) {
    // Wait for all workers if previous search is running.
    awaitWorkers();
    assert(checkAllVirtualLossIsZero(SearchTree->getRoot()));

    // Reset statistics.
    Stat.reset();

    // Setup the state to think.
    if (CQueue != nullptr) {
        CQueue->incrementGeneration();
    }
    Node* RootNode = SearchTree->updateRoot(*CurrentState);

    core::Move32 BestMove = core::Move32::MoveNone();

    // Start thinking.
#ifndef NDEBUG
    assert(!EQueue->isOpen());
    if (EQueue->count() != 0) {
        std::cerr << "[ERROR] EQueue->count() != 0 before search starts ("
                  << EQueue->count() << ")." << std::endl;
    }
    assert(EQueue->count() == 0);
#endif
    EQueue->open();
    FQueue->notifyEvaluationStarts();
    for (const auto& FeedWorker : FeedWorkers) {
        FeedWorker->start();
    }
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->start();
    }
    SWorkerMaster->enableImmediateLog();
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNode);
        SearchWorker->start();
    }

    {
        std::lock_guard<std::mutex> Lock(MutexStatus);
        Status = ManagerStatus::Thinking;
    }
    CVStatus.notify_all();

    // Await workers until the search stops.
    awaitWorkers();
    assert(checkAllVirtualLossIsZero(SearchTree->getRoot()));

#ifndef NDEBUG
    // Check the virtual loss of the root node is 0.
    if (SearchTree->getRoot() != nullptr) {
        const uint64_t RootVirtualLoss =
            SearchTree->getRoot()->getVisitsAndVirtualLoss() >>
            Node::VirtualLossShift;
        if (RootVirtualLoss != 0) {
            std::cerr << "[ERROR] Root's virtual loss after searching "
                         "check failed. Virtual loss: "
                      << RootVirtualLoss << std::endl;
        }
        assert(EQueue->count() == 0);
        assert(RootVirtualLoss == 0);
    }
#endif
    // Show statistics.
    if (PContext->printStatistics()) {
        PLogger->printStatistics(Stat);
    }

    if (STCallback != nullptr) {
        STCallback(SearchTree.get());
    }

    BestMove = getBestmove(RootNode);

    // Update the root node here for the garbage collectors
    // to release the previous root node.
    if (CQueue != nullptr) {
        CQueue->incrementGeneration();
    }
    CurrentState->doMove(BestMove);
    SearchTree->updateRoot(*CurrentState);

    if (CallCallback) {
        std::lock_guard<std::mutex> Lock(MutexStatus);

        // Start pondering before sending the bestmove
        // not to cause timing issue caused by pondering
        // and a given immediate next thinkNextMove() calling.
        if (Status == ManagerStatus::Thinking &&
            PContext->isPonderingEnabled() && !BestMove.isNone() &&
            !BestMove.isWin() && !checkMemoryBudgetForPondering()) {
            Node* RootNodePondering = SearchTree->getRoot();
            if (RootNodePondering->getPlyToTerminalSolved() == 0) {
                Status = ManagerStatus::Pondering;

                SWorkerMaster->setLimit(NoLimit);

                PLogger->setIsInverse(true);

                assert(!EQueue->isOpen());
                assert(EQueue->count() == 0);
                EQueue->open();
                FQueue->notifyEvaluationStarts();
                for (const auto& FeedWorker : FeedWorkers) {
                    FeedWorker->start();
                }
                for (const auto& EvaluationWorker : EvaluationWorkers) {
                    EvaluationWorker->start();
                }
                SWorkerMaster->disableImmediateLog();
                for (const auto& SearchWorker : SearchWorkers) {
                    SearchWorker->updateRoot(*CurrentState, *StateConfig,
                                             RootNodePondering);
                    SearchWorker->start();
                }
            }
        }

        if (Status != ManagerStatus::Pondering) {
            Status = ManagerStatus::Idle;
        }

        if (BestMoveCallback != nullptr) {
            BestMoveCallback(BestMove);
        }
    }
    CVStatus.notify_all();
}

void Manager::stopWorkers() {
    EQueue->close();
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->stop();
    }
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->stop();
    }
}

void Manager::awaitWorkers() {
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->await();
    }
    FQueue->notifyEvaluationStops();
    for (const auto& FeedWorker : FeedWorkers) {
        FeedWorker->stop();
    }
    for (const auto& FeedWorker : FeedWorkers) {
        FeedWorker->await();
    }
}

core::Move32 Manager::getBestmove(Node* Root) {
    if ((StateConfig->Rule & core::ER_Declare27) != 0) {
        if (CurrentState->canDeclare()) {
            return core::Move32::MoveWin();
        }
    }

    const auto* BestEdge = Root->mostPromisingEdge();

    if (BestEdge == nullptr) {
        return core::Move32::MoveNone();
    }

    return CurrentState->getMove32FromMove16(BestEdge->getMove());
}

bool Manager::checkMemoryBudgetForPondering() {
    if (NodeAllocator.getTotal() > 0 &&
        (double)NodeAllocator.getUsed() >
            (double)NodeAllocator.getTotal() * 0.8) {
        PLogger->printLog(
            "Pondering has been skipped due to little memory budget (Node).");
        return true;
    }

    if (EdgeAllocator.getTotal() > 0 &&
        (double)EdgeAllocator.getUsed() >
            (double)EdgeAllocator.getTotal() * 0.8) {
        PLogger->printLog(
            "Pondering has been skipped due to little memory budget (Edge).");
        return true;
    }

    return false;
}

void Manager::searchStopCallback() {
    stopWorkers();
}

bool Manager::checkAllVirtualLossIsZero(Node* Root) const {
    std::queue<Node*> Queue;
    Queue.push(Root);

    while (!Queue.empty()) {
        Node* N = Queue.front();
        Queue.pop();

        if (N == nullptr) {
            continue;
        }

        const uint64_t VirtualLoss =
            N->getVisitsAndVirtualLoss() >> Node::VirtualLossShift;

        if (VirtualLoss != 0) {
            return false;
        }

        const uint16_t NumChildren = N->getNumChildren();

        for (uint16_t I = 0; I < NumChildren; ++I) {
            Edge* E = &N->getEdge()[I];
            Node* Child = E->getTarget();

            if (Child != nullptr) {
                Queue.push(Child);
            }
        }
    }

    return true;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
