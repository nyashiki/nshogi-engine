//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "manager.h"
#include "../io/book.h"

#include <cmath>
#include <functional>

namespace nshogi {
namespace engine {
namespace mcts {

Manager::Manager(const Context* C, std::shared_ptr<logger::Logger> Logger)
    : PContext(C)
    , PLogger(std::move(Logger))
    , WakeUpSupervisor(false)
    , HasInterruptReceived(false)
    , IsExiting(false) {
    setupAllocator();
    setupGarbageCollector();
    setupMutexPool();
    setupSearchTree();
    setupCheckmateQueue(PContext->getNumCheckmateSearchThreads());
    setupCheckmateWorkers(PContext->getNumCheckmateSearchThreads());
    setupEvalCache(PContext->getEvalCacheMemoryMB());
    setupBook(PContext->isBookEnabled(), PContext->getBookPath());
    setupEvaluationQueue(PContext->getBatchSize(), PContext->getNumGPUs(),
                         PContext->getNumEvaluationThreadsPerGPU());
    setupEvaluationWorkers(PContext->getBatchSize(), PContext->getNumGPUs(),
                           PContext->getNumEvaluationThreadsPerGPU());
    setupSearchWorkers(PContext->getNumSearchThreads());
    setupSupervisor();
    setupWatchDog();

    PLogger->setIsNShogiExtensionLogEnabled(
        PContext->isNShogiExtensionLogEnabled());
}

Manager::~Manager() {
    interrupt();

    {
        std::lock_guard<std::mutex> LockS(MutexSupervisor);
        IsExiting = true;
    }
    CVSupervisor.notify_one();

    Supervisor->join();

    WatchdogWorker.reset(nullptr);
    for (auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker.reset(nullptr);
    }
    for (auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker.reset(nullptr);
    }
    for (auto& SearchWorker : SearchWorkers) {
        SearchWorker.reset(nullptr);
    }

    // SearchTree's destructor must be called before
    // GarbageCollector is released. Hence,
    // call the destructor explicitly here.
    SearchTree.reset(nullptr);
}

void Manager::thinkNextMove(
    const core::State& State, const core::StateConfig& Config,
    engine::Limit Lim,
    std::function<void(core::Move32, std::unique_ptr<ThoughtLog>)> Callback) {
    WatchdogWorker->stop();

    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    std::cerr << "[thinkNextMove()] await search worker ok ... " << std::endl;
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->await();
    }
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }
    WatchdogWorker->await();

    HasInterruptReceived.store(false);

    // Update the current state.
    {
        std::lock_guard<std::mutex> Lock(MutexSupervisor);
        CurrentState = std::make_unique<core::State>(State.clone());
        StateConfig = std::make_unique<core::StateConfig>(Config);
        Limit = std::make_unique<engine::Limit>(Lim);
        BestMoveCallback = Callback;
        WakeUpSupervisor = true;
        PLogger->setIsInverse(false);
    }

    // Wake up the supervisor and the watchdog.
    CVSupervisor.notify_one();
}

void Manager::interrupt() {
    HasInterruptReceived.store(true);

    WatchdogWorker->stop();
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->await();
    }
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }
    WatchdogWorker->await();
}

void Manager::setupAllocator() {
    NodeAllocator.resize((std::size_t)(
        0.1 * (double)(PContext->getAvailableMemoryMB() * 1024UL * 1024UL)));
    EdgeAllocator.resize((std::size_t)(
        0.9 * (double)(PContext->getAvailableMemoryMB() * 1024UL * 1024UL)));
}

void Manager::setupGarbageCollector() {
    GC = std::make_unique<GarbageCollector>(
        PContext->getNumGarbageCollectorThreads(), &NodeAllocator,
        &EdgeAllocator);
}

void Manager::setupMutexPool() {
    MtxPool = std::make_unique<MutexPool<lock::SpinLock>>(1000000);
}

void Manager::setupSearchTree() {
    SearchTree =
        std::make_unique<Tree>(GC.get(), &NodeAllocator, PLogger.get());
}

void Manager::setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs,
                                   std::size_t NumEvaluationWorkersPerGPU) {
    EQueue = std::make_unique<EvaluationQueue<global_config::FeatureType>>(
        BatchSize * NumGPUs * NumEvaluationWorkersPerGPU * 64);
}

void Manager::setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs,
                                     std::size_t NumEvaluationWorkersPerGPU) {
    std::size_t ThreadId = 0;
    for (std::size_t I = 0; I < NumGPUs; ++I) {
        for (std::size_t J = 0; J < NumEvaluationWorkersPerGPU; ++J) {
            EvaluationWorkers.emplace_back(
                std::make_unique<EvaluationWorker<global_config::FeatureType>>(
                    PContext, ThreadId, I, BatchSize, EQueue.get(),
                    ECache.get()));
            ++ThreadId;
        }
    }
}

void Manager::setupSearchWorkers(std::size_t NumSearchWorkers) {
    for (std::size_t I = 0; I < NumSearchWorkers; ++I) {
        SearchWorkers.emplace_back(
            std::make_unique<SearchWorker<global_config::FeatureType>>(
                &NodeAllocator, &EdgeAllocator, EQueue.get(), CQueue.get(),
                MtxPool.get(), ECache.get()));
    }
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
            std::make_unique<CheckmateWorker>(CQueue.get()));
    }
}

void Manager::setupEvalCache(std::size_t EvalCacheMB) {
    if (EvalCacheMB > 0) {
        ECache = std::make_unique<EvalCache>(EvalCacheMB);
    }
}

void Manager::setupBook(bool IsBookEnabled, const std::string& BookPath) {
    if (IsBookEnabled) {
        std::ifstream Ifs(BookPath);
        if (!Ifs) {
            PLogger->printLog("Can't open the book.");
            return;
        }

        const auto BookTemp = nshogi::engine::io::book::readBook(Ifs);
        for (const auto& BookEntry : BookTemp) {
            Book.emplace(BookEntry.huffmanCode(), BookEntry);
        }
        PLogger->printLog("Book loaded.");
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
    });
}

void Manager::setupWatchDog() {
    WatchdogWorker = std::make_unique<Watchdog>(PContext, &NodeAllocator,
                                                &EdgeAllocator, PLogger);
    WatchdogWorker->setStopSearchingCallback(
        std::bind(&Manager::watchdogStopCallback, this));
}

void Manager::doSupervisorWork(bool CallCallback) {
    // Setup the state to think.
    Node* RootNode = SearchTree->updateRoot(*CurrentState);

    core::Move32 BestMove = core::Move32::MoveNone();
    const auto BookEntryIt =
        Book.find(core::HuffmanCode::encode(CurrentState->getPosition()));

    std::unique_ptr<ThoughtLog> TL;
    if (BookEntryIt != Book.end() && CurrentState->getRepetitionStatus() ==
                                         core::RepetitionStatus::NoRepetition) {
        PLogger->printLog("Found a book move.");
        BestMove =
            CurrentState->getMove32FromMove16(BookEntryIt->second.bestMove());

        logger::PVLog Log;
        Log.WinRate = BookEntryIt->second.winRate();
        Log.DrawRate = BookEntryIt->second.drawRate();
        Log.PV.emplace_back(BestMove);
        PLogger->printPVLog(Log);
    } else {
        // Start thinking.
        assert(EQueue->count() == 0);
        EQueue->open();
        if (CQueue != nullptr) {
            CQueue->open();
        }
        for (const auto& SearchWorker : SearchWorkers) {
            SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNode);
            SearchWorker->start();
        }
        for (const auto& EvaluationWorker : EvaluationWorkers) {
            EvaluationWorker->start();
        }
        for (const auto& CheckmateWorker : CheckmateWorkers) {
            CheckmateWorker->start();
        }

        WatchdogWorker->updateRoot(CurrentState.get(), StateConfig.get(),
                                   RootNode);
        WatchdogWorker->setLimit(*Limit);
        WatchdogWorker->start();

        // Await workers until the search stops.
        for (const auto& SearchWorker : SearchWorkers) {
            SearchWorker->await();
        }
        for (const auto& EvaluationWorker : EvaluationWorkers) {
            EvaluationWorker->await();
        }
        for (const auto& CheckmateWorker : CheckmateWorkers) {
            CheckmateWorker->await();
        }
        WatchdogWorker->await();

        // Prepare ThoughtLog if IsThoughtLogEnabled is true.
        if (PContext->isThoughtLogEnabled()) {
            TL = std::make_unique<ThoughtLog>();
            const uint64_t NumChildren = RootNode->getNumChildren();
            TL->VisitCounts.reserve(NumChildren);
            for (std::size_t I = 0; I < NumChildren; ++I) {
                auto* Edge = &RootNode->getEdge()[I];
                auto* Child = Edge->getTarget();

                if (Child == nullptr) {
                    continue;
                }

                const uint64_t ChildVisits =
                    Child->getVisitsAndVirtualLoss() & Node::VisitMask;

                if (ChildVisits <= 1) {
                    continue;
                }

                TL->VisitCounts.emplace_back(Edge->getMove(), ChildVisits);
            }

            TL->WinRate = 0.0;
            TL->DrawRate = 0.0;
            const uint64_t Visits =
                RootNode->getVisitsAndVirtualLoss() & Node::VisitMask;
            if (Visits > 0) {
                TL->WinRate =
                    RootNode->getWinRateAccumulated() / (double)Visits;
                TL->DrawRate =
                    RootNode->getDrawRateAccumulated() / (double)Visits;
            }
        }

        BestMove = getBestmove(RootNode);
    }

    // Update the root node here for the garbage collectors
    // to release the previous root node.
    CurrentState->doMove(BestMove);
    SearchTree->updateRoot(*CurrentState);

    if (CallCallback) {
        // Start pondering before sending the bestmove
        // not to cause timing issue caused by pondering
        // and a given immediate next thinkNextMove() calling.
        if (PContext->isPonderingEnabled() && !BestMove.isNone() &&
            !BestMove.isWin() && !HasInterruptReceived.load() &&
            !checkMemoryBudgetForPondering()) {
            Node* RootNodePondering = SearchTree->getRoot();
            if (RootNodePondering->getPlyToTerminalSolved() == 0) {
                Limit = std::make_unique<engine::Limit>(NoLimit);

                PLogger->setIsInverse(true);

                EQueue->open();
                if (CQueue != nullptr) {
                    CQueue->open();
                }
                for (const auto& SearchWorker : SearchWorkers) {
                    SearchWorker->updateRoot(*CurrentState, *StateConfig,
                                             RootNodePondering);
                    SearchWorker->start();
                }
                for (const auto& EvaluationWorker : EvaluationWorkers) {
                    EvaluationWorker->start();
                }
                for (const auto& CheckmateWorker : CheckmateWorkers) {
                    CheckmateWorker->start();
                }

                WatchdogWorker->updateRoot(
                    CurrentState.get(), StateConfig.get(), RootNodePondering);
                WatchdogWorker->setLimit(*Limit);
                WatchdogWorker->start();
            }
        }

        if (BestMoveCallback != nullptr) {
            BestMoveCallback(BestMove, std::move(TL));
        }
    }
}

void Manager::stopWorkers() {
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->stop();
    }

    EQueue->close();
    for (const auto& EvaluationWorker : EvaluationWorkers) {
        EvaluationWorker->stop();
    }

    if (CQueue != nullptr) {
        CQueue->close();
    }
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->stop();
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

void Manager::watchdogStopCallback() {
    stopWorkers();
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
