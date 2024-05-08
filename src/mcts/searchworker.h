#ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
#define NSHOGI_ENGINE_MCTS_SEARCHWORKER_H

#include "checkmatesearcher.h"
#include "evalcache.h"
#include "tree.h"
#include "mutexpool.h"
#include "../evaluate/batch.h"
#include "../evaluate/evaluator.h"
#include "../evaluate/preset.h"
#include "../globalconfig.h"
#include "../lock/spinlock.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace mcts {

class SearchWorker {
 public:
    SearchWorker(std::size_t BatchSize, evaluate::Evaluator* Ev, CheckmateSearcher* CSearcher_ = nullptr, MutexPool<lock::SpinLock>* = nullptr, EvalCache* = nullptr);
    ~SearchWorker();

    void start(Node* Root, const core::State& St, const core::StateConfig& Config);
    void stop();
    void await();
    void join();

 private:
    struct LeafInfo {
        core::State State;
        Node* LeafNode;
    };

    const std::size_t BatchSizeMax;

    std::mutex Mtx;
    std::condition_variable Cv;
    std::condition_variable AwaitCv;

    evaluate::Batch<GlobalConfig::FeatureType> Batch;
    std::atomic<bool> IsRunning;
    bool IsFinnishing;

    CheckmateSearcher* CSearcher;

    Node* RootNode;
    MutexPool<lock::SpinLock>* MtxPool;
    EvalCache* ECache;
    EvalCache::EvalInfo EInfo;

    uint16_t RootPly;
    const core::StateConfig* StateConfig;

    std::thread WorkerThread;

    void mainLoop();
    void doOneIteration();

    template <bool PolicyLogits>
    void feedLeafNode(const LeafInfo& N, const float* Policy, float WinRate, float DrawRate);
    void evaluateStoredLeafNodes(std::size_t BatchIndexMax);
    bool evaluateByRule(const core::State& State, Node* N);

    void immediateUpdateByWin(Node* N) const;
    void immediateUpdateByLoss(Node* N) const;
    void immediateUpdateByDraw(Node* N) const;
    void updateAncestors(Node* N) const;
    void goBack(core::State* S) const;
    Node* selectLeafNode(core::State* S) const;

    Edge* computeUCBMaxEdge(const core::State& State, Node* N, bool RegardNotVisitedWin) const;
    void incrementVirtualLoss(Node* N) const;

    double computeWinRateOfChild(const core::State& State, Node* Child, uint64_t ChildVisits, uint64_t ChildVirtualVisits) const;

    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    std::vector<LeafInfo> LeafInfos;

    std::atomic<bool> IsRunningInternal_;
};


} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
