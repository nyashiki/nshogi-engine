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
    Worker(FrameQueue* QueueForSearch, FrameQueue* QueueForEvaluation, FrameQueue* QueueForSave, std::vector<core::Position>* InitialPositionsToPlay);

 private:
    bool doTask() override;

    SelfplayPhase initialize(Frame*) const;
    SelfplayPhase prepareRoot(Frame*) const;
    SelfplayPhase selectLeaf(Frame*) const;
    SelfplayPhase checkTerminal(Frame*) const;
    SelfplayPhase backpropagate(Frame*) const;
    SelfplayPhase sequentialHalving(Frame*) const;
    SelfplayPhase judge(Frame*) const;
    SelfplayPhase transition(Frame*) const;

    double sampleGumbelNoise() const;
    double transformQ(double, uint64_t MaxN) const;
    template <bool IsRoot>
    mcts::Edge* pickUpEdgeToExplore(Frame*, core::Color SideToMove, mcts::Node*) const;
    mcts::Edge* pickUpEdgeToExplore(Frame*, core::Color SideToMove, mcts::Node*, uint8_t Depth) const;
    double computeWinRateOfChild(Frame* F, core::Color SideToMove, mcts::Node* Child) const;
    bool isCheckmated(Frame* F) const;
    void sampleTopMMoves(Frame*) const;
    uint16_t executeSequentialHalving(Frame*) const;
    void updateSequentialHalvingSchedule(Frame*, uint16_t NumValidChilds) const;

    FrameQueue* FQueue;
    FrameQueue* EvaluationQueue;
    FrameQueue* SaveQueue;

    mutable std::mt19937_64 MT;

    std::vector<core::Position>* InitialPositions;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
