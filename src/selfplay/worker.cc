#include "worker.h"

#include <limits>
#include <stdexcept>
#include <cmath>

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/solver/dfs.h>

namespace nshogi {
namespace engine {
namespace selfplay {

Worker::Worker(FrameQueue* FQ, FrameQueue* EFQ, FrameQueue* SFQ, std::vector<core::Position>* InitialPositionsToPlay)
    : worker::Worker(true)
    , FQueue(FQ)
    , EvaluationQueue(EFQ)
    , SaveQueue(SFQ)
    , InitialPositions(InitialPositionsToPlay) {

    spawnThread();
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
        } else if (Task->getPhase() == SelfplayPhase::Save) {
            SaveQueue->add(std::move(Task));
        } else {
            FQueue->add(std::move(Task));
        }
    }

    return false;
}

SelfplayPhase Worker::initialize(Frame* F) const {
    // Setup a state.
    if (InitialPositions == nullptr || InitialPositions->size() == 0) {
        F->setState(std::make_unique<core::State>(core::StateBuilder::getInitialState()));
    } else {
        const core::Position& SampledPosition =
            InitialPositions->at(MT() % InitialPositions->size());
        F->setState(std::make_unique<core::State>(
                        core::StateBuilder::newState(SampledPosition)));
    }

    // Setup a config.
    auto Config = std::make_unique<core::StateConfig>();

    static std::uniform_int_distribution<> MaxPlyDistribution(160, 1024);
    static std::uniform_real_distribution<float> DrawRateDistribution(0.0f, 1.0f);

    Config->MaxPly = (uint16_t)MaxPlyDistribution(MT);

    uint64_t R = MT() % 4;
    Config->Rule = core::EndingRule::Declare27_ER;
    if (R < 2) {
        Config->BlackDrawValue = 0.5f;
        Config->WhiteDrawValue = 0.5f;
    } else {
        Config->BlackDrawValue = DrawRateDistribution(MT);
        Config->WhiteDrawValue = 1.0f - Config->BlackDrawValue;
    }

    F->setConfig(std::move(Config));

    // Other settings.
    F->setNumPlayouts(32);
    F->setNumSamplingMove(32);

    return SelfplayPhase::RootPreparation;
}

SelfplayPhase Worker::prepareRoot(Frame* F) const {
    // Update the search tree.
    F->getSearchTree()->updateRoot(*F->getState(), false);
    F->setRootPly(F->getState()->getPly());

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

    uint8_t Depth = 0;
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

        mcts::Edge* E = pickUpEdgeToExplore(F, Node, Depth);
        F->getState()->doMove(F->getState()->getMove32FromMove16(E->getMove()));

        if (E->getTarget() == nullptr) {
            auto NewNode = std::make_unique<mcts::Node>(Node);
            assert(NewNode != nullptr);

            E->setTarget(std::move(NewNode));
        }

        Node = E->getTarget();
        ++Depth;
    }

    F->setNodeToEvaluate(Node);
    return SelfplayPhase::LeafTerminalChecking;
}

SelfplayPhase Worker::checkTerminal(Frame* F) const {
    const auto RS = F->getState()->getRepetitionStatus();

    // Repetition.
    if (RS == core::RepetitionStatus::WinRepetition) {
        F->setEvaluation(nullptr, 1.0f, 0.0f);
        F->getNodeToEvalute()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::LossRepetition) {
        F->setEvaluation(nullptr, 0.0f, 0.0f);
        F->getNodeToEvalute()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::Repetition) {
        const float DrawValue = F->getState()->getSideToMove() == core::Black
                ? F->getStateConfig()->BlackDrawValue
                : F->getStateConfig()->WhiteDrawValue;
        F->setEvaluation(nullptr, DrawValue, 1.0f);
        F->getNodeToEvalute()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    }

    // Declaration.
    if (F->getStateConfig()->Rule == core::EndingRule::Declare27_ER) {
        if (F->getState()->canDeclare()) {
            F->setEvaluation(nullptr, 1.0f, 0.0f);
            return SelfplayPhase::Backpropagation;
        }
    }

    const auto LegalMoves = nshogi::core::MoveGenerator::generateLegalMoves(*F->getState());

    // Checkmate.
    if (LegalMoves.size() == 0) {
        if (F->getState()->getPly(false) > 0) {
            // Check if the last move is a checkmate by dropping a pawn.
            const auto LastMove = F->getState()->getLastMove();
            if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                F->setEvaluation(nullptr, 1.0f, 0.0f);
                return SelfplayPhase::Backpropagation;
            }
        }

        F->setEvaluation(nullptr, 0.0f, 0.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Max ply.
    if (F->getState()->getPly() >= F->getStateConfig()->MaxPly) {
        const float DrawValue = F->getState()->getSideToMove() == core::Black
                ? F->getStateConfig()->BlackDrawValue
                : F->getStateConfig()->WhiteDrawValue;
        F->setEvaluation(nullptr, DrawValue, 1.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Checkmate by search.
    if (F->getState()->getPly() > F->getRootPly() && isCheckmated(F)) {
        F->setEvaluation(nullptr, 0.0f, 0.0f);
        return SelfplayPhase::Backpropagation;
    }

    F->getNodeToEvalute()->expand(LegalMoves);
    return SelfplayPhase::Evaluation;
}

SelfplayPhase Worker::backpropagate(Frame* F) const {
    // Backpropagate win rate and draw rate.
    F->getNodeToEvalute()->updateAncestors<false>(
        F->getNodeToEvalute()->getWinRatePredicted(),
        F->getNodeToEvalute()->getDrawRatePredicted());

    // If the node is root node, extract top m moves sorted by
    // gumbel noise and policy.
    if (F->getState()->getPly() == F->getRootPly()) {
        F->getIsTarget().clear();
        F->getIsTarget().resize(F->getNodeToEvalute()->getNumChildren());

        // Set top m moves to explore by gumbel noise and policy.
        std::vector<std::pair<double, std::size_t>> ScoreWithIndex(F->getIsTarget().size());
        for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
            const mcts::Edge* Edge = F->getNodeToEvalute()->getEdge(I);
            ScoreWithIndex[I].first = F->getGumbelNoise().at(I) + Edge->getProbability();
            ScoreWithIndex[I].second = I;
        }

        const std::size_t NumSort = std::min(
                (std::size_t)F->getNumSamplingMove(), ScoreWithIndex.size());
        std::partial_sort(
                ScoreWithIndex.begin(),
                ScoreWithIndex.begin() + (long)NumSort,
                ScoreWithIndex.end(),
                [](const std::pair<double, std::size_t>& Elem1,
                   const std::pair<double, std::size_t>& Elem2) {
                    return Elem1.first > Elem2.first;
                });
        std::fill(F->getIsTarget().begin(), F->getIsTarget().end(), false);
        for (std::size_t I = 0; I < NumSort; ++I) {
            F->getIsTarget().at(ScoreWithIndex[I].second) = true;
        }
    }

    const uint64_t StopPlayout =
        std::min(F->getNumPlayouts() + 1, (uint64_t)F->getSearchTree()->getRoot()->getNumChildren() + 1);
    if (F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss() == StopPlayout) {
        // Add one above as one simulation was used for the root evaluation.
        return SelfplayPhase::Transition;
    }

    return SelfplayPhase::LeafSelection;
}

SelfplayPhase Worker::judge(Frame* F) const {
    const auto RS = F->getState()->getRepetitionStatus();

    if (RS == core::RepetitionStatus::WinRepetition) {
        F->setWinner(F->getState()->getSideToMove());
        return SelfplayPhase::Save;
    } else if (RS == core::RepetitionStatus::LossRepetition) {
        F->setWinner(~F->getState()->getSideToMove());
        return SelfplayPhase::Save;
    } else if (RS == core::RepetitionStatus::Repetition) {
        F->setWinner(core::NoColor);
        return SelfplayPhase::Save;
    }

    if (F->getStateConfig()->Rule == core::EndingRule::Declare27_ER &&
            F->getState()->canDeclare()) {
        F->setWinner(F->getState()->getSideToMove());
        return SelfplayPhase::Save;
    }

    if (core::MoveGenerator::generateLegalMoves(*F->getState()).size() == 0) {
        if (F->getState()->getPly(false) > 0) {
            // Check if the last move is a checkmate by dropping a pawn.
            const auto LastMove = F->getState()->getLastMove();
            if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                F->setWinner(F->getState()->getSideToMove());
                return SelfplayPhase::Save;
            }
        }

        F->setWinner(~F->getState()->getSideToMove());
        return SelfplayPhase::Save;
    }

    if (F->getState()->getPly() >= F->getStateConfig()->MaxPly) {
        F->setWinner(core::NoColor);
        return SelfplayPhase::Save;
    }

    return SelfplayPhase::RootPreparation;
}

SelfplayPhase Worker::transition(Frame* F) const {
    while (F->getState()->getPly() > F->getRootPly()) {
        F->getState()->undoMove();
    }

    double ScoreMax = std::numeric_limits<double>::lowest();
    mcts::Edge* ScoreMaxEdge = nullptr;

    uint64_t MaxN = 1;

    // for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
    //     if (!F->getIsTarget().at(I)) {
    //         continue;
    //     }
    //     mcts::Edge* Edge = F->getSearchTree()->getRoot()->getEdge(I);
    //     mcts::Node* Child = Edge->getTarget();
    //     assert(Child != nullptr);
    //     MaxN = std::max(MaxN, Child->getVisitsAndVirtualLoss());
    // }

    assert(F->getIsTarget().size() > 0);
    for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
        if (!F->getIsTarget().at(I)) {
            continue;
        }

        mcts::Edge* Edge = F->getSearchTree()->getRoot()->getEdge(I);
        mcts::Node* Child = Edge->getTarget();
        assert(Child != nullptr);

        const double Score =
            F->getGumbelNoise().at(I)
            + Edge->getProbability()
            + transformQ(computeWinRateOfChild(F, Child), MaxN);

        if (Score > ScoreMax) {
            ScoreMax = Score;
            ScoreMaxEdge = Edge;
        }
    }

    assert(ScoreMaxEdge != nullptr);
    F->getState()->doMove(F->getState()->getMove32FromMove16(ScoreMaxEdge->getMove()));
    return SelfplayPhase::Judging;
}

double Worker::sampleGumbelNoise() const {
    std::uniform_real_distribution<double> Distirbution(0.0, 1.0);
    const double U = Distirbution(MT);
    return -std::log(-std::log(U));
}

double Worker::transformQ(double Q, uint64_t MaxN) const {
    constexpr double C_VISIT = 50.0;
    constexpr double C_SCALE = 1.0;

    return (C_VISIT + (double)MaxN) * C_SCALE * Q;
}

mcts::Edge* Worker::pickUpEdgeToExplore(Frame* F, mcts::Node* N, uint8_t Depth) const {
    if (Depth == 0) {
        return pickUpEdgeToExploreAtRoot(F, N);
    }

    throw std::runtime_error("Not implemented yet.");
}

mcts::Edge* Worker::pickUpEdgeToExploreAtRoot(Frame* F, mcts::Node* N) const {
    mcts::Edge* EdgeToExplore = nullptr;
    double ScoreMax = std::numeric_limits<double>::lowest();

    const uint16_t NumChildren = N->getNumChildren();
    for (std::size_t I = 0; I < NumChildren; ++I) {
        if (!F->getIsTarget().at(I)) {
            // This edge is disabled (i.e., this node was not
            // chosen in the first sampling of m children).
            continue;
        }

        mcts::Edge* Edge = N->getEdge(I);
        mcts::Node* Child = Edge->getTarget();

        const double Score = (Child == nullptr)
            ? std::numeric_limits<double>::max()
            : (F->getGumbelNoise().at(I) + Edge->getProbability() + computeWinRateOfChild(F, Child));

        if (Score > ScoreMax) {
            ScoreMax = Score;
            EdgeToExplore = Edge;
        }
    }

    assert(EdgeToExplore != nullptr);
    return EdgeToExplore;
}

double Worker::computeWinRateOfChild(Frame* F, mcts::Node* Child) const {
    const uint64_t ChildVisits = Child->getVisitsAndVirtualLoss() & mcts::Node::VisitMask;
    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAcuumulated = Child->getDrawRateAccumulated();

    const double WinRate = ((double)ChildVisits - ChildWinRateAccumulated) / (double)ChildVisits;
    const double DrawRate = ChildDrawRateAcuumulated / (double)ChildVisits;

    const double DrawValue = (F->getState()->getSideToMove() == core::Black)
                                ? F->getStateConfig()->BlackDrawValue
                                : F->getStateConfig()->WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

bool Worker::isCheckmated(Frame* F) const {
    if (F->getState()->getCheckerBB().isZero()) {
        return false;
    }

    const auto Moves = core::MoveGenerator::generateLegalMoves(*F->getState());

    for (core::Move32 Move : Moves) {
        F->getState()->doMove(Move);
        const auto CheckmateMove = solver::dfs::solve(F->getState(), 3);
        F->getState()->undoMove();

        if (CheckmateMove.isNone()) {
            return false;
        }
    }

    return true;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
