#ifndef NSHOGI_ENGINE_SELFPLAY_FRAME_H
#define NSHOGI_ENGINE_SELFPLAY_FRAME_H

#include "phase.h"
#include "../mcts/tree.h"
#include <vector>
#include <memory>

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace selfplay {

struct Frame {
 public:
    Frame(mcts::GarbageCollector*);

    SelfplayPhase getPhase() const;
    core::State* getState();
    core::StateConfig* getStateConfig();
    mcts::Tree* getSearchTree();
    core::Color getWinner() const;

    void setPhase(SelfplayPhase);
    void setState(std::unique_ptr<core::State>&&);
    void setConfig(std::unique_ptr<core::StateConfig>&&);
    void setWinner(core::Color);

    mcts::Node* getNodeToEvalute();
    uint16_t getRootPly() const;
    void setNodeToEvaluate(mcts::Node*);
    void setRootPly(uint16_t);

    void setEvaluation(const float* Policy, float WinRate, float DrawRate);

    uint16_t getNumSamplingMove() const;
    std::vector<double>& getGumbelNoise();
    std::vector<bool>& getIsTarget();
    void setNumSamplingMove(uint16_t);

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

    // Gumbel.
    uint16_t NumSamplingMove;  // m in the paper.
    std::vector<double> GumbelNoise;
    std::vector<bool> IsTarget;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_FRAME_H
