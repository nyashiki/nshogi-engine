#ifndef NSHOGI_ENGINE_MCTS_LEAFCOLLECTOR_H
#define NSHOGI_ENGINE_MCTS_LEAFCOLLECTOR_H

#include "checkmatesearcher.h"
#include "evaluatequeue.h"
#include "edge.h"
#include "node.h"
#include "mutexpool.h"
#include "../lock/spinlock.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class LeafCollector {
 public:
    LeafCollector(EvaluationQueue<Features>*, MutexPool<lock::SpinLock>*, CheckmateSearcher*);

    void start();
    void stop();

    Node* collectOneLeaf();

 private:
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    void mainLoop();

    Edge* computeUCBMaxEdge(Node*, uint16_t NumChildren, bool regardNotVisitedWin);
    double computeWinRateOfChild(Node* Child, uint64_t ChildVisits, uint64_t ChildVirtualVisits);
    void incrementVirtualLosses(Node*);

    core::State State = core::StateBuilder::getInitialState();
    core::StateConfig Config;
    Node* RootNode;
    uint16_t RootPly;

    EvaluationQueue<Features>* EQueue;
    MutexPool<lock::SpinLock>* MtxPool;
    CheckmateSearcher* CSearcher;

    std::mutex Mutex;
    std::condition_variable CV;
    std::atomic<bool> IsRunnning;
    std::atomic<bool> IsThreadWorking;
    std::atomic<bool> IsExiting;

    std::thread Worker;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_LEAFCOLLECTOR_H
