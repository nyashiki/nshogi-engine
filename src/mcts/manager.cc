#include "manager.h"
#include "edge.h"
#include "garbagecollector.h"
#include "node.h"
#include "mutexpool.h"
#include "searchworkerbuilder.h"
#include "tree.h"
#include "../allocator/allocator.h"
#include "../globalconfig.h"

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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <cmath>

namespace nshogi {
namespace engine {
namespace mcts {

Manager::Manager(std::size_t NumGPUs, std::size_t NumSearchersPerGPU, std::size_t NumCheckmateSearchers, std::size_t BatchSize, logger::Logger* Logger)
    : GC(GlobalConfig::getConfig().getNumGarbageCollectorThreads()), SearchTree(&GC, Logger), WatchDogEnabled(false), WatchDogRunning(false), WatchDogFinishing(false), PLogger(Logger) {
    if (NumGPUs <= 0) {
        std::cerr << "NumGPUs must be greater than or equal to 1." << std::endl;
        std::exit(1);
    }
    if (NumSearchersPerGPU <= 0) {
        std::cerr << "NumSearchersPerGPU must be greater than or equal to 1." << std::endl;
        std::exit(1);
    }

    MtxPool = std::make_unique<MutexPool<lock::SpinLock>>(1000000);
    if (GlobalConfig::getConfig().getEvalCacheMemoryMB() > 0) {
        ECache = std::make_unique<EvalCache>(GlobalConfig::getConfig().getEvalCacheMemoryMB());
    }

    if (NumCheckmateSearchers > 0) {
        CSearcher = std::make_unique<CheckmateSearcher>(5, NumCheckmateSearchers);
    }

    for (std::size_t GPUId = 0; GPUId < NumGPUs; ++GPUId) {
        for (std::size_t I = 0; I < NumSearchersPerGPU; ++I) {
#if defined(EXECUTOR_ZERO)
            Infers.emplace_back(std::make_unique<infer::Zero>());
#elif defined(EXECUTOR_NOTHING)
            Infers.emplace_back(std::make_unique<infer::Nothing>());
#elif defined(EXECUTOR_RANDOM)
            Infers.emplace_back(std::make_unique<infer::Random>(0));
#elif defined(EXECUTOR_TRT)
            auto TRT = std::make_unique<infer::TensorRT>(GPUId, BatchSize, GlobalConfig::FeatureType::size());
            TRT->load(GlobalConfig::getConfig().getWeightPath(), true);
            Infers.emplace_back(std::move(TRT));
#endif

            Evaluators.emplace_back(std::make_unique<evaluate::Evaluator>(BatchSize, Infers.back().get()));
            // SearchWorkers.emplace_back(std::make_unique<SearchWorker>(BatchSize, Evaluators.back().get(), CSearcher.get(), MtxPool.get(), ECache.get()));
            SearchWorkers.emplace_back(
                SearchWorkerBuilder()
                    .setBatchSize(BatchSize)
                    .setEvaluator(Evaluators.back().get())
                    .setCheckmateSearcher(CSearcher.get())
                    .setMutexPool(MtxPool.get())
                    .setEvalCache(ECache.get())
                    .build());
        }
    }

    WatchDogThread = std::make_unique<std::thread>([Logger, this]() {
        while (true) {
            if (!WatchDogEnabled) {
                WatchDogRunning = false;
            }

            {
                std::unique_lock<std::mutex> Lock(WatchDogMtx);

                WatchDogCv.wait(Lock, [&]() {
                    return WatchDogEnabled.load() || WatchDogFinishing;
                });

                if (WatchDogFinishing) {
                    break;
                }
            }

            WatchDogRunning = true;

            // Here, watch dog is running.
            const auto StartTime = std::chrono::steady_clock::now();
            auto PrevLogTime = StartTime;
            const uint64_t StartingNumNodes = SearchTree.getRoot()->getVisitsAndVirtualLoss() & 0xFFFFFFFFULL;

            while (WatchDogEnabled) {
                const auto CurrentTime = std::chrono::steady_clock::now();
                const uint32_t Elapsed = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(CurrentTime - StartTime).count());
                const uint32_t LogElapsed = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(CurrentTime - PrevLogTime).count());

                // Output thinking log.
                if (LogElapsed >= GlobalConfig::getConfig().getLogMargin()) {
                    dumpLog(StartingNumNodes, Elapsed);
                    PrevLogTime = CurrentTime;
                }

                // The position is solved.
                if (SearchTree.getRoot()->getPlyToTerminalSolved() != 0) {
                    if (WatchDogEnabled) {
                        if (Logger != nullptr) {
                            dumpLog(StartingNumNodes, Elapsed);
                            Logger->printLog("Solved.");
                        }
                        stop(false);
                    }
                    break;
                }

                // Time limit.
                if (isThinkingTimeOver(Elapsed)) {
                    if (WatchDogEnabled) {
                        if (Logger != nullptr) {
                            dumpLog(StartingNumNodes, Elapsed);
                            Logger->printLog("Time limit.");
                        }
                        stop(false);
                    }
                    break;
                }

                // Memory limit.
                { // (a) Node limit.
                    const auto& NodeAllocator = allocator::getNodeAllocator();
                    const double Factor = GlobalConfig::getConfig().getMemoryLimitFactor();

                    if (NodeAllocator.getTotal() > 0 &&
                            (double)NodeAllocator.getUsed() > (double)NodeAllocator.getTotal() * Factor) {
                        if (WatchDogEnabled) {
                            if (Logger != nullptr) {
                                dumpLog(StartingNumNodes, Elapsed);
                                Logger->printLog("Memory limit (Node).");
                            }
                            stop(false);
                        }
                        break;
                    }

                    const auto& EdgeAllocator = allocator::getEdgeAllocator();
                    if (EdgeAllocator.getTotal() > 0 &&
                            (double)EdgeAllocator.getUsed() > (double)EdgeAllocator.getTotal() * Factor) {
                        if (WatchDogEnabled) {
                            if (Logger != nullptr) {
                                dumpLog(StartingNumNodes, Elapsed);
                                Logger->printLog("Memory limit (Edge).");
                            }
                            stop(false);
                        }
                        break;
                    }
                }

                // Judge whether continue searching or make a dicision.
                bool DidMakeUpMind = didMakeUpMind(Elapsed);
                if (DidMakeUpMind) {
                    if (WatchDogEnabled) {
                        if (Logger != nullptr) {
                            dumpLog(StartingNumNodes, Elapsed);
                            Logger->printLog("Made up mind.");
                        }
                        stop(false);
                    }
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
    });
}

Manager::~Manager() {
    stop();

    // Stop the watch dog.
    {
        std::lock_guard<std::mutex> Lock(WatchDogMtx);
        WatchDogFinishing = true;
    }

    WatchDogCv.notify_one();
    WatchDogThread->join();

    for (std::size_t I = 0; I < SearchWorkers.size(); ++I) {
        SearchWorkers[I]->join();
        Evaluators[I].reset(nullptr);
        Infers[I].reset(nullptr);
    }
}

void Manager::thinkNextMove(const core::State& State, const core::StateConfig& Config, const engine::Limit& Limit, void (*CallBack)(const core::Move32& Move)) {
    Node* RootNode = SearchTree.updateRoot(State);
    ConfigInternal = Config;
    CurrentColor = State.getSideToMove();

    VisitsPre.clear();
    ElapsedPre = 0;
    BestEdgePre = nullptr;
    MakeUpMindCount = 0;
    MakeUpMindWinRateMean = 0.0;
    MakeUpMindWinRateVar = 0.0;

    CallBackFunctionPtr = CallBack;

    std::atomic<bool> AllWorkerStarted = false;

    Thread = std::make_unique<std::thread>([RootNode, &State, &AllWorkerStarted, this]() {
        if (CSearcher != nullptr) {
            CSearcher->start();
        }

        for (auto& Worker : SearchWorkers) {
            Worker->start(RootNode, State, ConfigInternal);
        }

        AllWorkerStarted.store(true);

        for (auto& Worker : SearchWorkers) {
            Worker->await();
        }

        if (CSearcher != nullptr) {
            CSearcher->stop();
        }

        {
            const uint64_t VisitsAndVirtualLoss = RootNode->getVisitsAndVirtualLoss();
            const uint64_t Visits = VisitsAndVirtualLoss & Node::VisitMask;
            const uint64_t VirtualLoss = VisitsAndVirtualLoss >> Node::VirtualLossShift;
            const double WinRate = RootNode->getWinRateAccumulated() / (double)Visits;
            const double DrawRate = RootNode->getDrawRateAccumulated() / (double)Visits;

            assert(VirtualLoss == 0);

            std::stringstream ss;
            ss << "[Root] Visit: " << Visits << ", VirtualLoss: " << VirtualLoss
                    << ", WinRate: " << WinRate << ", DrawRate: " << DrawRate;
            if (PLogger != nullptr) {
                PLogger->printLog(ss.str().c_str());
            }
        }

        if (CallBackFunctionPtr != nullptr) {
            BestMove = [&]() {
                if ((ConfigInternal.Rule & core::Declare27_ER) != 0) {
                    if (State.canDeclare()) {
                        return core::Move32::MoveWin();
                    }
                }

                const mcts::Edge* BestEdge = RootNode->mostPromisingEdge();

                if (BestEdge == nullptr) {
                    return core::Move32::MoveNone();
                } else {
                    return State.getMove32FromMove16(BestEdge->getMove());
                }
            }();
        }
    });

    LimitInternal = Limit;
    WatchDogEnabled = true;

    while (!AllWorkerStarted.load()) {
        std::this_thread::yield();
    }

    WatchDogCv.notify_one();
}

logger::PVLog Manager::getPV() const {
    logger::PVLog Log;

    Node* N = SearchTree.getRoot();
    const uint32_t Visits = (uint32_t)(SearchTree.getRoot()->getVisitsAndVirtualLoss());

    if (Visits == 0) {
        return Log;
    }

    Log.SolvedGameEndPly = N->getPlyToTerminalSolved();
    Log.WinRate = N->getWinRateAccumulated() / Visits;
    Log.DrawRate = N->getDrawRateAccumulated() / Visits;
    Log.DrawValue = (CurrentColor == core::Black)? ConfigInternal.BlackDrawValue : ConfigInternal.WhiteDrawValue;

    Log.NumNodes = Visits;

    while (N != nullptr) {
        Edge* E = N->mostPromisingEdge();

        if (E == nullptr) {
            break;
        }

        Log.PV.push_back(E->getMove());

        N = E->getTarget();
    }

    return Log;
}

void Manager::dumpLog(uint64_t StartingNumNodes, uint32_t Elapsed) const {
    if (PLogger == nullptr) {
        return;
    }

    logger::PVLog Log = getPV();
    Log.ElapsedMilliSeconds = Elapsed;
    if (Elapsed > 0) {
        Log.NodesPerSecond = (uint64_t)((double)((Log.NumNodes - StartingNumNodes) * 1000ULL) / (double)Elapsed);
    }

    PLogger->printPVLog(Log);
}

void Manager::stop(bool WaitUntilWatchDogStops) {
    WatchDogEnabled = false;

    if (WaitUntilWatchDogStops) {
        while (WatchDogRunning) {
            std::this_thread::yield();
        }
    }

    // This function can be called at a time by
    // more than one threads so we need mutex here.
    std::lock_guard<std::mutex> Lock(StopMtx);

    if (Thread == nullptr) {
        return;
    }

    for (auto& Worker : SearchWorkers) {
        Worker->stop();
    }

    if (CSearcher != nullptr) {
        CSearcher->stop();
    }

    Thread->join();
    Thread.reset();

    if (CallBackFunctionPtr != nullptr) {
        CallBackFunctionPtr(BestMove);
    }
}

bool Manager::isThinkingTimeOver(uint32_t Elapsed) const {
    if (LimitInternal.isNoLimit()) {
        return false;
    }

    const int32_t ThinkingTimeMaximum = [&]() {
        int32_t ThinkingTime = (int32_t)LimitInternal.TimeLimitMilliSeconds;
        if (LimitInternal.ByoyomiMilliSeconds > 0) {
            ThinkingTime += (int32_t)LimitInternal.ByoyomiMilliSeconds;
        }

        if (LimitInternal.IncreaseMilliSeconds > 0) {
            ThinkingTime += (int32_t)LimitInternal.IncreaseMilliSeconds;
        }

        return ThinkingTime;
    }() - (int32_t)Elapsed;

    if (ThinkingTimeMaximum <= (int32_t)GlobalConfig::getConfig().getThinkingTimeMargin()) {
        return true;
    }

    return false;
}

bool Manager::didMakeUpMind(uint32_t Elapsed) const {
    if (LimitInternal.isNoLimit()) {
        return false;
    }

    Node* RootNode = SearchTree.getRoot();

    if (RootNode == nullptr) {
        return false;
    }

    if (Elapsed < ElapsedPre + 470) {
        return false;
    }

    const uint16_t NumChildren = RootNode->getNumChildren();

    uint64_t VisitsSum = 0;

    std::vector<double> Visits(NumChildren, 0.0);
    for (uint16_t I = 0; I < NumChildren; ++I) {
        Edge* E = RootNode->getEdge((std::size_t)I);
        Node* Child = E->getTarget();

        if (Child != nullptr) {
            const uint64_t VisitsAndVirtualLoss = Child->getVisitsAndVirtualLoss();
            const uint64_t V = VisitsAndVirtualLoss & Node::VisitMask;

            VisitsSum += V;
            Visits[I] = (double)(V);
        }
    }

    for (uint16_t I = 0; I < NumChildren; ++I) {
        Visits[I] /= (double)VisitsSum;
    }

    Edge* BestEdge = RootNode->mostPromisingEdge();

    if (BestEdge == BestEdgePre) {
        if (NumChildren == (uint16_t)(VisitsPre.size())) {
            double KLDivergence = 0.0;
            double KLDivergenceToPredicted = 0.0;

            for (std::size_t I = 0; I < Visits.size(); ++I) {
                if (VisitsPre[I] == 0) {
                    continue;
                } else if (Visits[I] == 0) {
                    KLDivergence = std::numeric_limits<double>::max();
                    break;
                }

                const double Predicted = RootNode->getEdge(I)->getProbability();
                const double D = VisitsPre[I] * std::log(VisitsPre[I] / Visits[I]);
                const double D2 = Predicted * std::log(Predicted / Visits[I]);

                KLDivergence += D;
                KLDivergenceToPredicted += D2;
            }

            const double KLDivergenceThreshold =
                (KLDivergenceToPredicted < 0.4) ? 1e-5 : 1e-6;
            if (KLDivergence < KLDivergenceThreshold) {
                return true;
            }
        }
    }

    VisitsPre.swap(Visits);
    ElapsedPre = Elapsed;
    BestEdgePre = BestEdge;

    return false;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
