#include "manager.h"

#ifdef EXECUTOR_ZERO

#include "../infer/zero.h"

#endif

#ifdef EXECUTOR_NOTHING

#include "../infer/nothing.h"

#endif

#ifdef EXECUTOR_RANDOM

#include "../infer/random.h"

#endif

#ifdef EXECUTOR_TRT

#include "../infer/trt.h"

#endif

#include <functional>
#include <cmath>

namespace nshogi {
namespace engine {
namespace mcts {

Manager::Manager(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumSearchWorkers, std::size_t NumEvaluationWorkersPerGPU, std::shared_ptr<logger::Logger> Logger)
    : PLogger(std::move(Logger))
    , WakeUpSupervisor(false)
    , IsPonderingEnabled(false)
    , IsExiting(false) {
    setupGarbageCollector();
    setupMutexPool();
    setupSearchTree();
    setupEvaluationQueue(BatchSize, NumGPUs, NumEvaluationWorkersPerGPU);
    setupEvaluationWorkers(BatchSize, NumGPUs, NumEvaluationWorkersPerGPU);
    setupSearchWorkers(NumSearchWorkers);
    setupSupervisor();
    setupWatchDog();
}

Manager::~Manager() {
    stopWorkers();

    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->await();
    }

    {
        std::lock_guard<std::mutex> LockS(MutexSupervisor);
        IsExiting = true;
    }
    CVSupervisor.notify_one();

    Supervisor->join();
}

void Manager::setIsPonderingEnabled(bool Value) {
    IsPonderingEnabled = Value;
}

void Manager::thinkNextMove(const core::State& State, const core::StateConfig& Config, const engine::Limit& Lim, void (*CallBack)(const core::Move32&)) {
    WatchdogWorker->stop();

    std::cerr << "[thinkNextMove()] await ... " << std::endl;
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->await();
    }
    std::cerr << "[thinkNextMove()] await ... ok." << std::endl;
    WatchdogWorker->await();

    // Update the current state.
    {
        std::lock_guard<std::mutex> Lock(MutexSupervisor);
        CurrentState = std::make_unique<core::State>(State.clone());
        StateConfig = std::make_unique<core::StateConfig>(Config);
        Limit = std::make_unique<engine::Limit>(Lim);
        BestmoveCallback = CallBack;
        WakeUpSupervisor = true;
        PLogger->setIsInverse(false);
    }
    std::cerr << "[thinkNextMove()] update the current state ... ok." << std::endl;

    // Wake up the supervisor and the watchdog.
    CVSupervisor.notify_one();
}

void Manager::interrupt() {
    WatchdogWorker->stop();
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->await();
    }
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->await();
    }
    WatchdogWorker->await();
}

void Manager::setupGarbageCollector() {
    GC = std::make_unique<GarbageCollector>(
        GlobalConfig::getConfig().getNumGarbageCollectorThreads());
}

void Manager::setupMutexPool() {
    MtxPool = std::make_unique<MutexPool<lock::SpinLock>>(1000000);
}

void Manager::setupSearchTree() {
    SearchTree = std::make_unique<Tree>(GC.get(), PLogger.get());
}

void Manager::setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU) {
    EQueue = std::make_unique<EvaluationQueue<GlobalConfig::FeatureType>>(
            BatchSize * NumGPUs * NumEvaluationWorkersPerGPU * 2);
}

void Manager::setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU) {
    for (std::size_t I = 0; I < NumGPUs; ++I) {
        for (std::size_t J = 0; J < NumEvaluationWorkersPerGPU; ++J) {
#if defined(EXECUTOR_ZERO)
            Infers.emplace_back(std::make_unique<infer::Zero>());
#elif defined(EXECUTOR_NOTHING)
            Infers.emplace_back(std::make_unique<infer::Nothing>());
#elif defined(EXECUTOR_RANDOM)
            Infers.emplace_back(std::make_unique<infer::Random>(0));
#elif defined(EXECUTOR_TRT)
            auto TRT = std::make_unique<infer::TensorRT>(0, BatchSize, GlobalConfig::FeatureType::size());
            TRT->load(GlobalConfig::getConfig().getWeightPath(), true);
            Infers.emplace_back(std::move(TRT));
#endif
            Evaluators.emplace_back(
                    std::make_unique<evaluate::Evaluator>(
                        BatchSize, Infers.back().get()));

            EvaluateWorkers.emplace_back(
                    std::make_unique<EvaluateWorker<GlobalConfig::FeatureType>>(
                        BatchSize, EQueue.get(), Evaluators.back().get()));
        }
    }
}

void Manager::setupSearchWorkers(std::size_t NumSearchWorkers) {
    for (std::size_t I = 0; I < NumSearchWorkers; ++I) {
        SearchWorkers.emplace_back(std::make_unique<SearchWorker<GlobalConfig::FeatureType>>(
            EQueue.get(), MtxPool.get(), nullptr));
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
    WatchdogWorker = std::make_unique<Watchdog>(PLogger);
    WatchdogWorker->setStopSearchingCallback(std::bind(&Manager::watchdogStopCallback, this));
}

void Manager::doSupervisorWork(bool CallCallback) {
    // Setup the state to think.
    Node* RootNode = SearchTree->updateRoot(*CurrentState);

    // Start thinking.
    std::cerr << "[doSupervisorWork()] start workers ..." << std::endl;
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNode);
        SearchWorker->start();
    }
    for (const auto& EvaluateWorker : EvaluateWorkers) {
        EvaluateWorker->start();
    }
    std::cerr << "[doSupervisorWork()] start workers ... ok." << std::endl;

    WatchdogWorker->updateRoot(CurrentState.get(), StateConfig.get(), RootNode);
    WatchdogWorker->setLimit(*Limit);
    std::cerr << "[doSupervisorWork()] start watchdog ..." << std::endl;
    if (WatchdogWorker->getIsRunning()) {
        std::cerr << "[doSupervisorWork()] ERROR !!!!!!!!!!!!!!! WATCHDOG IS RUNNING." << std::endl;
    }
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
    std::cerr << "[doSupervisorWork()] await watchdog ..." << std::endl;
    WatchdogWorker->await();
    std::cerr << "[doSupervisorWork()] await watchdog ... ok." << std::endl;

    if (CallCallback) {
        const auto Bestmove = getBestmove(RootNode);

        // Start pondering before sending the bestmove
        // not to cause timing issue caused by pondering
        // and a given immediate next thinkNextMove() calling.
        if (IsPonderingEnabled && !Bestmove.isNone() && !Bestmove.isWin()) {
            CurrentState->doMove(Bestmove);
            Node* RootNodePondering = SearchTree->updateRoot(*CurrentState);
            Limit = std::make_unique<engine::Limit>(NoLimit);

            // Start pondering.
            std::cerr << "[doSupervisorWork()] start pondering ..." << std::endl;
            for (const auto& SearchWorker : SearchWorkers) {
                SearchWorker->updateRoot(*CurrentState, *StateConfig, RootNodePondering);
                SearchWorker->start();
            }
            for (const auto& EvaluateWorker : EvaluateWorkers) {
                EvaluateWorker->start();
            }

            WatchdogWorker->updateRoot(CurrentState.get(), StateConfig.get(), RootNodePondering);
            WatchdogWorker->setLimit(*Limit);
            WatchdogWorker->start();

            std::cerr << "[doSupervisorWork()] start pondering ... ok." << std::endl;
        }

        BestmoveCallback(Bestmove);
    }
}

void Manager::stopWorkers() {
    for (const auto& SearchWorker : SearchWorkers) {
        SearchWorker->stop();
    }
    for (const auto& EvaluationWorker : EvaluateWorkers) {
        EvaluationWorker->stop();
    }
}

core::Move32 Manager::getBestmove(Node* Root) {
    if ((StateConfig->Rule & core::Declare27_ER) != 0) {
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

void Manager::watchdogStopCallback() {
    std::cerr << "[watchdogStopCallback()] got callback ..." << std::endl;
    stopWorkers();
    std::cerr << "[watchdogStopCallback()] got callback ... ok." << std::endl;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
