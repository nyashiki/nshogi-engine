#ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
#define NSHOGI_ENGINE_MCTS_SEARCHWORKER_H

#include "checkmatesearcher.h"
#include "evaluatequeue.h"
#include "edge.h"
#include "node.h"
#include "mutexpool.h"
#include "../lock/spinlock.h"
#include "../worker/worker.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class SearchWorker : public worker::Worker {
 public:
    SearchWorker(EvaluationQueue<Features>*, MutexPool<lock::SpinLock>*, CheckmateSearcher*);
    ~SearchWorker();

    void updateRoot(const core::State&, const core::StateConfig&, Node*);
    Node* collectOneLeaf();
    bool expandLeaf(Node*);

    void evaluateByRule(Node*);

    void immediateUpdateByWin(Node*);
    void immediateUpdateByLoss(Node*);
    void immediateUpdateByDraw(Node*);
    void immediateUpdate(Node*);

 private:
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    bool doTask() override;

    Edge* computeUCBMaxEdge(Node*, uint16_t NumChildren, bool regardNotVisitedWin);
    double computeWinRateOfChild(Node* Child, uint64_t ChildVisits, uint64_t ChildVirtualVisits);
    void incrementVirtualLosses(Node*);

    std::unique_ptr<core::State> State;
    core::StateConfig Config;
    Node* RootNode;
    uint16_t RootPly;

    EvaluationQueue<Features>* EQueue;
    MutexPool<lock::SpinLock>* MtxPool;
    CheckmateSearcher* CSearcher;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKER_H
