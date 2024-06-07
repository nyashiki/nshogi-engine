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
    Worker(FrameQueue*, FrameQueue*);

 private:
    bool doTask() override;

    SelfplayPhase initialize(Frame*) const;
    SelfplayPhase prepareRoot(Frame*) const;
    SelfplayPhase selectLeaf(Frame*) const;
    SelfplayPhase checkTerminal(Frame*) const;
    SelfplayPhase backpropagate(Frame*) const;
    SelfplayPhase judge(Frame*) const;

    double sampleGumbelNoise() const;
    mcts::Edge* pickUpEdgeToExplore(mcts::Node*) const;

    FrameQueue* FQueue;
    FrameQueue* EvaluationQueue;

    std::mt19937_64 MT;

    std::vector<core::Position> InitialPositions;
    std::shared_ptr<mcts::GarbageCollector> GarbageCollector;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
