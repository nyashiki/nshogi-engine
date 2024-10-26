#include "manager.h"

#include <functional>
#include <cmath>

namespace nshogi {
namespace engine {
namespace mcts {

Manager::Manager(const Context* C, std::shared_ptr<logger::Logger> Logger)
    : PContext(C)
    , PLogger(std::move(Logger))
    , WakeUpSupervisor(false)
    , HasInterruptReceived(false)
    , IsPonderingEnabled(false)
    , IsExiting(false) {
    setupGarbageCollector();
    setupMutexPool();
    setupSearchTree();
    setupCheckmateQueue(PContext->getNumCheckmateSearchThreads());
    setupCheckmateWorkers(PContext->getNumCheckmateSearchThreads());
    setupEvalCache(PContext->getEvalCacheMemoryMB());
    setupEvaluationQueue(PContext->getBatchSize(), PContext->getNumGPUs(), PContext->getNumEvaluationThreadsPerGPU());
    setupEvaluationWorkers(PContext->getBatchSize(), PContext->getNumGPUs(), PContext->getNumEvaluationThreadsPerGPU());
    setupSearchWorkers(PContext->getNumSearchThreads());
    setupSupervisor();
    setupWatchDog();
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
    for (auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker.reset(nullptr);
    }
    for (auto& SearchWorker : SearchWorkers) {
        SearchWorker.reset(nullptr);
    }
}

void Manager::setIsPonderingEnabled(bool Value) {
    IsPonderingEnabled = Value;
}

void Manager::thinkNextMove(const core::State& State, const core::StateConfig& Config, engine::Limit Lim, std::function<void(core::Move32)> Callback) {
    WatchdogWorker->stop();

    std::cerr << "[thinkNextMove()] await ... " << std::endl;
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    std::cerr << "[thinkNextMove()] await search worker ok ... " << std::endl;
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->await();
    }
    std::cerr << "[thinkNextMove()] await evaluation worker ok ... " << std::endl;
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }
    std::cerr << "[thinkNextMove()] await checkmate worker ... ok." << std::endl;
    WatchdogWorker->await();
    std::cerr << "[thinkNextMove()] await ... ok." << std::endl;

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
    std::cerr << "[thinkNextMove()] update the current state ... ok." << std::endl;

    // Wake up the supervisor and the watchdog.
    CVSupervisor.notify_one();
}

void Manager::interrupt() {
    HasInterruptReceived.store(true);

    WatchdogWorker->stop();
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->await();
    }
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }
    WatchdogWorker->await();
}

void Manager::setupGarbageCollector() {
    GC = std::make_unique<GarbageCollector>(
        PContext->getNumGarbageCollectorThreads());
}

void Manager::setupMutexPool() {
    MtxPool = std::make_unique<MutexPool<lock::SpinLock>>(1000000);
}

void Manager::setupSearchTree() {
    SearchTree = std::make_unique<Tree>(GC.get(), PLogger.get());
}

void Manager::setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU) {
    EQueue = std::make_unique<EvaluationQueue<global_config::FeatureType>>(
            BatchSize * NumGPUs * NumEvaluationWorkersPerGPU * 2);
}

void Manager::setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU) {
    for (std::size_t I = 0; I < NumGPUs; ++I) {
        for (std::size_t J = 0; J < NumEvaluationWorkersPerGPU; ++J) {
            EvaluateWorkers.emplace_back(
                    std::make_unique<EvaluateWorker<global_config::FeatureType>>(
                        PContext, I, BatchSize, EQueue.get(), ECache.get()));
        }
    }
}

void Manager::setupSearchWorkers(std::size_t NumSearchWorkers) {
    for (std::size_t I = 0; I < NumSearchWorkers; ++I) {
        SearchWorkers.emplace_back(std::make_unique<SearchWorker<global_config::FeatureType>>(
            EQueue.get(), CQueue.get(), MtxPool.get(), ECache.get()));
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
        CheckmateWorkers.emplace_back(std::make_unique<CheckmateWorker>(CQueue.get()));
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

                CVSupervisor.wait(Lock, [this]() {
                    return WakeUpSupervisor || IsExiting;
                });

                std::cerr << "[setupSupervisor()] the supervisor has been woken up." << std::endl;

                if (IsExiting) {
                    break;
                }
            }

            std::cerr << "[setupSupervisor()] doSupervisorWork() ..." << std::endl;
            doSupervisorWork(true);
            std::cerr << "[setupSupervisor()] doSupervisorWork() ... ok." << std::endl;

            {
                std::lock_guard<std::mutex> Lock(MutexSupervisor);
                WakeUpSupervisor = false;
            }
        }
    });
}

void Manager::setupWatchDog() {
    WatchdogWorker = std::make_unique<Watchdog>(PContext, PLogger);
    WatchdogWorker->setStopSearchingCallback(std::bind(&Manager::watchdogStopCallback, this));
}

void Manager::doSupervisorWork(bool CallCallback) {
    // Setup the state to think.
    Node* RootNode = SearchTree->updateRoot(*CurrentState);

    // Start thinking.
    std::cerr << "[doSupervisorWork()] start workers ..." << std::endl;
    assert(EQueue->count() == 0);
    EQueue->open();
    if (CQueue != nullptr) {
        CQueue->open();
    }
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNode);
        SearchWorker->start();
    }
    for (const auto& EvaluateWorker : EvaluateWorkers) {
        EvaluateWorker->start();
    }
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->start();
    }

    std::cerr << "[doSupervisorWork()] start workers ... ok." << std::endl;

    WatchdogWorker->updateRoot(CurrentState.get(), StateConfig.get(), RootNode);
    WatchdogWorker->setLimit(*Limit);
    std::cerr << "[doSupervisorWork()] start watchdog ..." << std::endl;
    WatchdogWorker->start();
    std::cerr << "[doSupervisorWork()] start watchdog ... ok." << std::endl;

    // Await workers until the search stops.
    std::cerr << "[doSupervisorWork()] await workers ..." << std::endl;
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    std::cerr << "[doSupervisorWork()] await search workers ... ok." << std::endl;
    for (const auto& EvaluateWorker : EvaluateWorkers) {
        EvaluateWorker->await();
    }
    std::cerr << "[doSupervisorWork()] await evaluation workers ... ok." << std::endl;
    for (const auto& CheckmateWorker : CheckmateWorkers) {
        CheckmateWorker->await();
    }
    std::cerr << "[doSupervisorWork()] await watchdog ..." << std::endl;
    WatchdogWorker->await();
    std::cerr << "[doSupervisorWork()] await watchdog ... ok." << std::endl;

    // Update the root node here for the garbage collectors
    // to release the previous root node.
    const auto BestMove = getBestmove(RootNode);
    CurrentState->doMove(BestMove);
    SearchTree->updateRoot(*CurrentState);

    if (CallCallback) {
        // Start pondering before sending the bestmove
        // not to cause timing issue caused by pondering
        // and a given immediate next thinkNextMove() calling.
        if (IsPonderingEnabled && !BestMove.isNone() && !BestMove.isWin() && !HasInterruptReceived.load() && !checkMemoryBudgetForPondering()) {
            Node* RootNodePondering = SearchTree->getRoot();
            if (RootNodePondering->getPlyToTerminalSolved() == 0) {
                Limit = std::make_unique<engine::Limit>(NoLimit);

                PLogger->setIsInverse(true);

                // Start pondering.
                std::cerr << "[doSupervisorWork()] start pondering ..." << std::endl;
                EQueue->open();
                if (CQueue != nullptr) {
                    CQueue->open();
                }
                for (const auto& SearchWorker : SearchWorkers) {
                    SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNodePondering);
                    SearchWorker->start();
                }
                for (const auto& EvaluateWorker : EvaluateWorkers) {
                    EvaluateWorker->start();
                }
                for (const auto& CheckmateWorker : CheckmateWorkers) {
                    CheckmateWorker->start();
                }

                WatchdogWorker->updateRoot(CurrentState.get(), StateConfig.get(), RootNodePondering);
                WatchdogWorker->setLimit(*Limit);
                WatchdogWorker->start();

                std::cerr << "[doSupervisorWork()] start pondering ... ok." << std::endl;
            }
        }

        if (BestMoveCallback != nullptr) {
            BestMoveCallback(BestMove);
        }
    }
}

void Manager::stopWorkers() {
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->stop();
    }

    EQueue->close();
    for (const auto& EvaluationWorker : EvaluateWorkers) {
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
    const auto& NodeAllocator = allocator::getNodeAllocator();

    if (NodeAllocator.getTotal() > 0 &&
            (double)NodeAllocator.getUsed() > (double)NodeAllocator.getTotal() * 0.6) {
        PLogger->printLog("Pondering has been skipped due to little memory budget (Node).");
        return true;
    }

    const auto& EdgeAllocator = allocator::getEdgeAllocator();
    if (EdgeAllocator.getTotal() > 0 &&
            (double)EdgeAllocator.getUsed() > (double)EdgeAllocator.getTotal() * 0.6) {
        PLogger->printLog("Pondering has been skipped due to little memory budget (Edge).");
        return true;
    }

    return false;
}

void Manager::watchdogStopCallback() {
    std::cerr << "[watchdogStopCallback()] got callback ..." << std::endl;
    stopWorkers();
    std::cerr << "[watchdogStopCallback()] got callback ... ok." << std::endl;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
