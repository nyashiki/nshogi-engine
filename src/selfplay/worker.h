#ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
#define NSHOGI_ENGINE_SELFPLAY_WORKER_H

#include "framequeue.h"
#include "../worker/worker.h"

#include <random>
#include <vector>

#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace selfplay {

class Worker : public worker::Worker {
 public:
    Worker(FrameQueue* QueueForSearch, FrameQueue* QueueForEvaluation, FrameQueue* QueueForSave);

 private:
    bool doTask() override;

    SelfplayPhase initialize(Frame*) const;
    SelfplayPhase prepareRoot(Frame*) const;
    SelfplayPhase selectLeaf(Frame*) const;
    SelfplayPhase checkTerminal(Frame*) const;
    SelfplayPhase backpropagate(Frame*) const;
    SelfplayPhase judge(Frame*) const;
    SelfplayPhase transition(Frame*) const;

    double sampleGumbelNoise() const;
    mcts::Edge* pickUpEdgeToExplore(Frame*, mcts::Node*, uint8_t Depth) const;
    mcts::Edge* pickUpEdgeToExploreAtRoot(Frame*, mcts::Node*) const;
    double computeWinRateOfChild(Frame* F, mcts::Node* Child) const;

    FrameQueue* FQueue;
    FrameQueue* EvaluationQueue;
    FrameQueue* SaveQueue;

    mutable std::mt19937_64 MT;

    std::vector<core::Position> InitialPositions;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
