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
    Frame();

    SelfplayPhase getPhase();
    core::State* getState();
    core::StateConfig* getStateConfig();
    mcts::Tree* getSearchTree();

    void setPhase(SelfplayPhase);
    void setSearchTree(std::unique_ptr<mcts::Tree>&&);
    void setState(std::unique_ptr<core::State>&&);
    void setConfig(std::unique_ptr<core::StateConfig>&&);

    mcts::Node* getNodeToEvalute();
    uint16_t getRootPly() const;
    void setNodeToEvaluate(mcts::Node*);
    void allocatePolicyArray();
    void setRootPly(uint16_t);

    float* getPolicyPredicted();
    float getWinRatePredicted();
    float getDrawRatePredicted();
    void setPolicyPredicted(float*);
    void setWinRatePredicted(float);
    void setDrawRatePredicted(float);

    uint64_t getCurrentPlayouts() const;
    std::vector<double>& getGumbelNoise();

 private:
    SelfplayPhase Phase;

    // Game.
    std::unique_ptr<core::State> State;
    std::unique_ptr<core::StateConfig> StateConfig;

    // Search.
    uint16_t RootPly;
    std::unique_ptr<mcts::Tree> SearchTree;
    mcts::Node* NodeToEvaluate;

    // Evaluation.
    std::unique_ptr<float[]> PolicyPredicted;
    float WinRatePredicted;
    float DrawRatePredicted;

    // Gumbel.
    uint64_t NumPlayouts; // n in the paper.
    uint16_t NumSamplingMove;  // m in the paper.
    uint64_t NumCurrentPlayouts;
    std::vector<double> GumbelNoise;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_FRAME_H
