#include "frame.h"

#include <nshogi/ml/common.h>
#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace selfplay {

Frame::Frame(mcts::GarbageCollector* GC)
    : Phase(SelfplayPhase::Initialization) {
    setSearchTree(std::make_unique<mcts::Tree>(GC, nullptr));
    allocatePolicyArray();
    GumbelNoise.resize(600);
}

SelfplayPhase Frame::getPhase() const {
    return Phase;
}

core::State* Frame::getState() {
    return State.get();
}

core::StateConfig* Frame::getStateConfig() {
    return StateConfig.get();
}

mcts::Tree* Frame::getSearchTree() {
    return SearchTree.get();
}

void Frame::setPhase(SelfplayPhase SP) {
    Phase = SP;
}

void Frame::setSearchTree(std::unique_ptr<mcts::Tree>&& Tree) {
    SearchTree = std::move(Tree);
}

void Frame::allocatePolicyArray() {
    LegalPolicyLogits = std::make_unique<float[]>(ml::MoveIndexMax);
}

core::Color Frame::getWinner() const {
    return Winner;
}

void Frame::setState(std::unique_ptr<core::State>&& S) {
    State = std::move(S);
}

void Frame::setConfig(std::unique_ptr<core::StateConfig>&& SC) {
    StateConfig = std::move(SC);
}

void Frame::setWinner(core::Color C) {
    Winner = C;
}

mcts::Node* Frame::getNodeToEvalute() {
    return NodeToEvaluate;
}

uint16_t Frame::getRootPly() const {
    return RootPly;
}

void Frame::setNodeToEvaluate(mcts::Node* N) {
    NodeToEvaluate = N;
}

void Frame::setRootPly(uint16_t Ply) {
    RootPly = Ply;
}

void Frame::setEvaluation(const float* Policy, float WinRate, float DrawRate) {
    const uint16_t NumChildren = NodeToEvaluate->getNumChildren();
    assert(Policy != nullptr || NumChildren == 0);

    for (uint16_t I = 0; I < NumChildren; ++I) {
        const std::size_t MoveIndex =
            ml::getMoveIndex(State->getSideToMove(), NodeToEvaluate->getEdge(I)->getMove());
        LegalPolicyLogits[I] = Policy[MoveIndex];
    }

    if (NodeToEvaluate != SearchTree->getRoot()) {
        ml::math::softmax_(LegalPolicyLogits.get(), NumChildren, 1.0f);
    }

    NodeToEvaluate->setEvaluation(LegalPolicyLogits.get(), WinRate, DrawRate);
}

uint64_t Frame::getNumPlayouts() const {
    return NumPlayouts;
}

uint64_t Frame::getSequentialHalvingPlayouts() const {
    return SequentialHalvingPlayouts;
}

uint8_t Frame::getSequentialHalvingCount() const {
    return SequentialHalvingCount;
}

uint16_t Frame::getNumSamplingMove() const {
    return NumSamplingMove;
}

std::vector<double>& Frame::getGumbelNoise() {
    return GumbelNoise;
}

std::vector<bool>& Frame::getIsTarget() {
    return IsTarget;
}

void Frame::setNumPlayouts(uint64_t N) {
    NumPlayouts = N;
}

void Frame::setSequentialHalvingPlayouts(uint64_t N) {
    SequentialHalvingPlayouts = N;
}

void Frame::setSequentialHalvingCount(uint8_t C) {
    SequentialHalvingCount = C;
}

void Frame::setNumSamplingMove(uint16_t M) {
    NumSamplingMove = M;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
