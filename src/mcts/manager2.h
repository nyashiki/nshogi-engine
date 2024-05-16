#ifndef NSHOGI_ENGINE_MCTS_MANAGER2_H
#define NSHOGI_ENGINE_MCTS_MANAGER2_H

#include "evaluateworker.h"
#include "leafcollector.h"
#include "evaluatequeue.h"
#include "garbagecollector.h"
#include "tree.h"
#include "watchdog.h"
#include "../limit.h"
#include "../globalconfig.h"
#include "../evaluate/preset.h"
#include "../evaluate/evaluator.h"
#include "../infer/infer.h"

#include <condition_variable>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

class Manager2 {
 public:
    Manager2(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumSearchWorkers, std::size_t NumEvaluationWorkersPerGPU, std::shared_ptr<logger::Logger> Logger);
    ~Manager2();

    void setIsPonderingEnabled(bool Value);

    void thinkNextMove(const core::State&, const core::StateConfig&, const engine::Limit&, void (*CallBack)(const core::Move32&));
    void interrupt();

 private:
    void setupGarbageCollector();
    void setupMutexPool();
    void setupSearchTree();
    void setupEvaluationQueue(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU);
    void setupEvaluationWorkers(std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumEvaluationWorkersPerGPU);
    void setupSearchWorkers(std::size_t NumSearchWorkers);
    void setupSupervisor();
    void setupWatchDog();

    void doSupervisorWork(bool CallCackback);
    void doWatchdogWork();

    void stopWorkers();
    core::Move32 getBestmove(Node* Root);

    void watchdogStopCallback();

    std::unique_ptr<Tree> SearchTree;
    std::unique_ptr<GarbageCollector> GC;
    std::unique_ptr<EvaluationQueue<GlobalConfig::FeatureType>> EQueue;
    std::unique_ptr<MutexPool<lock::SpinLock>> MtxPool;
    std::vector<std::unique_ptr<LeafCollector<GlobalConfig::FeatureType>>> LeafCollectors;
    std::vector<std::unique_ptr<EvaluateWorker<GlobalConfig::FeatureType>>> EvaluateWorkers;

    std::vector<std::unique_ptr<infer::Infer>> Infers;
    std::vector<std::unique_ptr<evaluate::Evaluator>> Evaluators;

    std::shared_ptr<logger::Logger> PLogger;

    bool WakeUpSupervisor;
    std::unique_ptr<std::thread> Supervisor;
    std::mutex MutexSupervisor;
    std::condition_variable CVSupervisor;

    std::unique_ptr<Watchdog> WatchdogWorker;

    std::unique_ptr<core::State> CurrentState;
    std::unique_ptr<core::StateConfig> StateConfig;
    std::unique_ptr<engine::Limit> Limit;
    void (*BestmoveCallback)(const core::Move32&);
    bool IsPonderingEnabled;

    bool IsExiting;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_MANAGER2_H
