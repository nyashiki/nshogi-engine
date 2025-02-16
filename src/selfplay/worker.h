//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
#define NSHOGI_ENGINE_SELFPLAY_WORKER_H

#include "../allocator/allocator.h"
#include "../mcts/evalcache.h"
#include "../worker/worker.h"
#include "framequeue.h"
#include "selfplayinfo.h"
#include "shogi816k.h"

#include <random>
#include <vector>

#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace selfplay {

class Worker : public worker::Worker {
 public:
    Worker(FrameQueue* QueueForSearch, FrameQueue* QueueForEvaluation,
           FrameQueue* QueueForSave, allocator::Allocator* NodeAllocator,
           allocator::Allocator* EdgeAllocator, mcts::EvalCache*,
           std::vector<core::Position>* InitialPositionsToPlay,
           bool UseShogi816k, SelfplayInfo*);

 private:
    bool doTask() override;

    SelfplayPhase initialize(Frame*);
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
    mcts::Edge* pickUpEdgeToExplore(Frame*, core::Color SideToMove,
                                    mcts::Node*) const;
    mcts::Edge* pickUpEdgeToExplore(Frame*, core::Color SideToMove, mcts::Node*,
                                    uint8_t Depth) const;
    double computeWinRate(Frame* F, core::Color SideToMove,
                          mcts::Node* Child) const;
    double computeWinRateOfChild(Frame* F, core::Color SideToMove,
                                 mcts::Node* Child) const;
    bool isCheckmated(Frame* F) const;
    void sampleTopMMoves(Frame*) const;
    uint16_t executeSequentialHalving(Frame*) const;
    void updateSequentialHalvingSchedule(Frame*, uint16_t NumValidChilds) const;

    FrameQueue* FQueue;
    FrameQueue* EvaluationQueue;
    FrameQueue* SaveQueue;

    allocator::Allocator* NA;
    allocator::Allocator* EA;

    mcts::EvalCache* EvalCache;

    mutable std::mt19937_64 MT;

    std::vector<core::Position>* InitialPositions;

    const bool USE_SHOGI816K;
    PositionBuilderShogi816k PositionBuilder;

    SelfplayInfo* SInfo;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_WORKER_H
