#include "worker.h"

#include <nshogi/core/movegenerator.h>

#include <cmath>

namespace nshogi {
namespace engine {
namespace selfplay {

Worker::Worker(FrameQueue* FQ, FrameQueue* EFQ)
    : worker::Worker(true)
    , FQueue(FQ)
    , EvaluationQueue(EFQ) {
}

bool Worker::doTask() {
    auto Tasks = FQueue->getAll();

    while (!Tasks.empty()) {
        auto Task = std::move(Tasks.front());
        Tasks.pop();

        assert(Task->getPhase() != SelfplayPhase::Evaluation);

        if (Task->getPhase() == SelfplayPhase::Initialization) {
            Task->setPhase(initialize(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::RootPreparation) {
            Task->setPhase(prepareRoot(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::LeafSelection) {
            Task->setPhase(selectLeaf(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::LeafTerminalChecking) {
            Task->setPhase(checkTerminal(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::Backpropagation) {
            Task->setPhase(backpropagate(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::Judging) {
            Task->setPhase(judge(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::Transition) {
            Task->setPhase(transition(Task.get()));
        }

        if (Task->getPhase() == SelfplayPhase::Evaluation) {
            EvaluationQueue->add(std::move(Task));
        } else {
            FQueue->add(std::move(Task));
        }
    }

    return false;
}

SelfplayPhase Worker::initialize(Frame* F) const {
    // Prepare components.
    if (F->getSearchTree() == nullptr) {
        F->setSearchTree(std::make_unique<mcts::Tree>(GarbageCollector.get(), nullptr));
    }

    if (F->getPolicyPredicted() == nullptr) {
        F->allocatePolicyArray();
    }

    // Setup a state.
    if (InitialPositions.size() == 0) {
        F->setState(std::make_unique<core::State>(core::StateBuilder::getInitialState()));
    } else {
        const Position& SampledPosition =
            InitialPositions.at(MT() % InitialPositions.size());
        F->setState(std::make_unique<core::State>(SampledPosition));
    }

    // Setup a config.
    auto Config = std::make_unique<core::StateConfig>();

    static std::uniform_int_distribution<> MaxPlyDistribution(160, 1024);
    static std::uniform_real_distribution<float> DrawRateDistribution(0.0f, 1.0f);

    Config->MaxPly = MaxPlyDistribution(MT);

    uint64_t R = MT() % 4;
    Config->Rule = core::EndingRule::Declare27_ER;
    if (R < 2) {
        Config->BlackDrawValue = 0.5f;
        Config->WhiteDrawValue = 0.5f;
    } else {
        Config->BlackDrawValue = DrawRateDistribution(MT);
        Config->WhiteDrawValue = 1.0f - Config->BlackDrawValue;
    }

    return SelfplayPhase::RootPreparation;
}

SelfplayPhase Worker::prepareRoot(Frame* F) const {
    assert(F->getCurrentPlayouts() == 0);

    // Update the search tree.
    F->getSearchTree()->updateRoot(F->getState(), false);
    F->setRootPly(F->getState().getPly());

    // Sample gumbel noises.
    // We don't care about actual the number of legal moves.
    // In stead, we prepare sufficient enough number of gumbel noises.
    for (std::size_t I = 0; I < F->getGumbelNoise().size(); ++I) {
        F->getGumbelNoise().at(I) = sampleGumbelNoise();
    }

    return SelfplayPhase::LeafSelection;
}

SelfplayPhase Worker::selectLeaf(Frame* F) const {
    assert(F->getState()->getPly() >= F->getRootPly());

    while (F->getState()->getPly() > F->getRootPly()) {
        F->getState()->undoMove();
    }

    mcts::Node* Node = F->getSearchTree()->getRoot();
    assert(Node->getRepetitionStatus() == core::RepetitionStatus::NoRepetition);

    while (true) {
        if (Node->getVisitsAndVirtualLoss() == 0) {
            break;
        }

        if (Node->getNumChildren() == 0) {
            break;
        }

        if (Node->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
            break;
        }

        mcts::Edge* E = pickUpEdgeToExplore(Node);
        F->getState()->doMove(F->getState()->getMove32FromMove16(E->getMove()));

        if (E->getTarget() == nullptr) {
            auto NewNode = std::make_unique<mcts::Node>(Node);
            assert(NewNode != nullptr);

            E->setTarget(std::move(NewNode));
        }

        Node = E->getTarget();
    }

    F->setNodeToEvaluate(Node);
    return SelfplayPhase::LeafTerminalChecking;
}

SelfplayPhase Worker::checkTerminal(Frame* F) const {
    const auto RS = State->getRepetitionStatus();
    F->getNodeToEvalute()->setRepetitionStatus(RS);

    // Repetition.
    if (RS == core::RepetitionStatus::WinRepetition ||
            RS == core::RepetitionStatus::SuperiorRepetition) {
        F->setWinRatePredicted(1.0f);
        F->setDrawRatePredicted(0.0f);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::LossRepetition ||
            RS == core::RepetitionStatus::InferiorRepetition) {
        F->setWinRatePredicted(0.0f);
        F->setDrawRatePredicted(0.0f);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::Repetition) {
        F->setWinRatePredicted(
                F->getState()->getSideToMove() == core::Black ?
                    F->getStateConfig()->BlackDrawValue : F->getStateConfig()->WhiteDrawValue);
        F->setDrawRatePredicted(1.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Declaration.
    if (F->getStateConfig()->Rule == core::EndingRule::Declare27_ER) {
        if (F->getState()->canDeclare()) {
            F->setWinRatePredicted(1.0f);
            F->setDrawRatePredicted(0.0f);
            return SelfplayPhase::Backpropagation;
        }
    }

    // Checkmate.
    if (core::MoveGenerator::generateLegalMoves(*F->getState()).size() == 0) {
        F->setWinRatePredicted(0.0f);
        F->setDrawRatePredicted(0.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Max ply.
    if (F->getState()->getPly() >= F->getStateConfig()->MaxPly) {
        F->setWinRatePredicted(
                F->getState()->getSideToMove() == core::Black ?
                    F->getStateConfig()->BlackDrawValue : F->getStateConfig()->WhiteDrawValue);
        F->setDrawRatePredicted(1.0f);
        return SelfplayPhase::Backpropagation;
    }

    return SelfplayPhase::Evaluation;
}

SelfplayPhase Worker::backpropagate(Frame* F) const {
    // Generate the legal moves.
    const auto LegalMoves = nshogi::core::generateLegalMoves(F->getState());

    // Update the node.
    F->getNodeToEvalute()->expand(LegalMoves);

    // Set policy, win rate, and draw rate.
    F->getNodeToEvalute()->setEvaluation(F->getPolicyPredicted(),
            F->getWinRatePredicted(), F->getDrawRatePredicted());

    // Backpropagate win rate and draw rate.
    F->getNodeToEvalute()->updateAncestors(
            F->getWinRatePredicted(), F->getDrawRatePredicted());

    return SelfplayPhase::LeafSelection;
}

double Worker::sampleGumbelNoise() const {
    std::uniform_real_distribution<double> Distirbution(0.0, 1.0);
    const double U = Distirbution(MT);
    return -std::log(-std::log(U));
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
