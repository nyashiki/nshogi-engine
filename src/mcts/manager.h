#ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
#define NSHOGI_ENGINE_MCTS_MANAGER_H

#include "checkmatesearcher.h"
#include "edge.h"
#include "evalcache.h"
#include "mutexpool.h"
#include "searchworker.h"
#include "garbagecollector.h"
#include "../allocator/default.h"
#include "../evaluate/evaluator.h"
#include "../limit.h"
#include "../logger//logger.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <nshogi/core/stateconfig.h>
#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace mcts {

class Manager {
 public:
    Manager(std::size_t NumGPUs, std::size_t NumSearchersPerGPU, std::size_t NumCheckmateSearchers, std::size_t BatchSize, logger::Logger* Logger = nullptr);
    ~Manager();

    void thinkNextMove(const core::State& State, const core::StateConfig& Config, const engine::Limit& Limit, void (*CallBack)(const core::Move32& Move));

    void stop(bool WaitUntilWatchDogStops = true);

 private:
    logger::PVLog getPV() const;
    void dumpLog(uint64_t StartingNumNodes, uint32_t Elapsed) const;
    bool isThinkingTimeOver(uint32_t Elapsed) const;
    bool didMakeUpMind(uint32_t Elapsed) const;

    GarbageCollector GC;
    Tree SearchTree;
    std::unique_ptr<CheckmateSearcher> CSearcher;

    std::unique_ptr<std::thread> Thread;

    std::unique_ptr<std::thread> WatchDogThread;
    std::atomic<bool> WatchDogEnabled;
    std::atomic<bool> WatchDogRunning;
    bool WatchDogFinishing;
    std::mutex WatchDogMtx;
    std::mutex StopMtx;
    std::condition_variable WatchDogCv;

    std::vector<std::unique_ptr<SearchWorker>> SearchWorkers;
    std::vector<std::unique_ptr<infer::Infer>> Infers;
    std::vector<std::unique_ptr<evaluate::Evaluator>> Evaluators;
    std::unique_ptr<MutexPool> MtxPool;
    std::unique_ptr<EvalCache> ECache;
    core::StateConfig ConfigInternal;
    core::Color CurrentColor;
    Limit LimitInternal;

    mutable std::vector<double> VisitsPre;
    mutable uint32_t ElapsedPre;
    mutable Edge* BestEdgePre;
    mutable uint64_t MakeUpMindCount;
    mutable double MakeUpMindWinRateMean;
    mutable double MakeUpMindWinRateVar;

    core::Move32 BestMove;
    void (*CallBackFunctionPtr)(const core::Move32&) = nullptr;
    logger::Logger* PLogger;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_MCTS_MANAGER_H
