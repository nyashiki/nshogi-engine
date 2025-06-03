//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_SELFPLAY_FRAME_H
#define NSHOGI_ENGINE_SELFPLAY_FRAME_H

#include "../allocator/allocator.h"
#include "../mcts/evalcache.h"
#include "../mcts/tree.h"
#include "phase.h"
#include <memory>
#include <vector>

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace selfplay {

struct Frame {
 public:
    Frame(mcts::GarbageCollector*, allocator::Allocator* NodeAllocator);

    SelfplayPhase getPhase() const;
    core::State* getState();
    core::StateConfig* getStateConfig();
    mcts::Tree* getSearchTree();
    core::Color getWinner() const;

    void setPhase(SelfplayPhase);
    void setState(std::unique_ptr<core::State>&&);
    void setConfig(std::unique_ptr<core::StateConfig>&&);
    void setWinner(core::Color);

    mcts::Node* getNodeToEvaluate();
    uint16_t getRootPly() const;
    void setNodeToEvaluate(mcts::Node*);
    void setRootPly(uint16_t);

    void setEvaluationCache(mcts::EvalCache*);
    template <bool Aggregated>
    void setEvaluation(const float* Policy, float WinRate, float DrawRate);

    uint64_t getNumPlayouts() const;
    uint64_t getSequentialHalvingPlayouts() const;
    uint8_t getSequentialHalvingCount() const;
    uint16_t getNumSamplingMoves() const;
    std::vector<double>& getGumbelNoise();
    std::vector<bool>& getIsTarget();
    void setNumPlayouts(uint64_t);
    void setSequentialHalvingPlayouts(uint64_t);
    void setSequentialHalvingCount(uint8_t);
    void setNumSamplingMoves(uint16_t);

 private:
    void setSearchTree(std::unique_ptr<mcts::Tree>&&);
    void allocatePolicyArray();

    SelfplayPhase Phase;

    // Game.
    std::unique_ptr<core::State> State;
    std::unique_ptr<core::StateConfig> StateConfig;

    // Result.
    core::Color Winner;

    // Search.
    uint16_t RootPly;
    std::unique_ptr<mcts::Tree> SearchTree;
    mcts::Node* NodeToEvaluate;
    std::unique_ptr<float[]> LegalPolicyLogits;
    mcts::EvalCache* EvalCache;

    // Gumbel.
    uint64_t NumPlayouts; // n in the paper.
    uint64_t SequentialHalvingPlayouts;
    uint8_t SequentialHalvingCount;
    uint16_t NumSamplingMoves; // m in the paper.
    std::vector<double> GumbelNoise;
    std::vector<bool> IsTarget;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_FRAME_H
