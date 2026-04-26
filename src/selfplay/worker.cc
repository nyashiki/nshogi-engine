//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "worker.h"
#include "../mcts/pointer.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/sfen.h>
#include <nshogi/ml/math.h>
#include <nshogi/solver/dfs.h>

namespace nshogi {
namespace engine {
namespace selfplay {

Worker::Worker(FrameQueue* FQ, FrameQueue* EFQ, FrameQueue* SFQ,
               allocator::Allocator* NodeAllocator,
               allocator::Allocator* EdgeAllocator, mcts::EvalCache* EC,
               uint64_t NumPlayouts, uint16_t NumSamplingMoves,
               double FullSearchRatio,
               std::vector<core::Position>* InitialPositionsToPlay,
               bool UseShogi816k, SelfplayInfo* SI)
    : worker::Worker(true)
    , FQueue(FQ)
    , EvaluationQueue(EFQ)
    , SaveQueue(SFQ)
    , NA(NodeAllocator)
    , EA(EdgeAllocator)
    , EvalCache(EC)
    , MyNumPlayouts(NumPlayouts)
    , MyNumSamplingMoves(NumSamplingMoves)
    , MyFullSearchRatio(FullSearchRatio)
    , InitialPositions(InitialPositionsToPlay)
    , USE_SHOGI816K(UseShogi816k)
    , Solver(64)
    , SInfo(SI) {

    std::random_device SeedGen;
    MT.seed(SeedGen());

    spawnThread();
}

bool Worker::doTask() {
    auto Tasks = FQueue->getAll();

    while (!Tasks.empty()) {
        auto Task = std::move(Tasks.front());
        Tasks.pop();

        assert(Task->getPhase() != SelfplayPhase::Evaluation);

        //
        // clang-format off
        //
        // [selfplay loop]
        // intiialize --> prepareRoot --> selectLeaf --> checkTerminal --> evaluation --> backpropagate --> sequentialHalving -- (if search is done) --> transition -------> judge -- (if the game is finished) --> save
        //     |               |              |                |                                    |              |                                  (proceed one move)       |                                      |
        //     |               |              |                |                                    |              |                                                           |                                      |
        //     |               |              |                -> (if the leaf node is a terminal) ->              |                                                           |                                      |
        //     |               |              |                                                                    |                                                           |                                      |
        //     |               |              --<----------------------------------------------------------------<--                                                           |                                      |
        //     |               |                                                                                                                                               |                                      |
        //     |               --<-------------------------------------------------------------------------------------------------------------------------------------------<--                                      |
        //     |                                                                                                                                                                                                      |
        //     --<--------------------------------------------------------------------------------------------- (start a new game) ---------------------------------------------------------------------------------<--
        //
        // clang-format on
        //

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
        } else if (Task->getPhase() == SelfplayPhase::SequentialHalving) {
            Task->setPhase(sequentialHalving(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::Transition) {
            Task->setPhase(transition(Task.get()));
        } else if (Task->getPhase() == SelfplayPhase::Judging) {
            Task->setPhase(judge(Task.get()));
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

SelfplayPhase Worker::initialize(Frame* F) {
    // Setup a state.
    if (USE_SHOGI816K) {
        F->setState(std::make_unique<core::State>(
            core::StateBuilder::newState(PositionBuilder.build())));
    } else if (InitialPositions == nullptr || InitialPositions->size() == 0) {
        F->setState(std::make_unique<core::State>(
            core::StateBuilder::getInitialState()));
    } else {
        if (MT() % 100 == 0) {
            F->setState(std::make_unique<core::State>(
                core::StateBuilder::getInitialState()));
        } else {
            const core::Position& SampledPosition =
                InitialPositions->at(MT() % InitialPositions->size());
            F->setState(std::make_unique<core::State>(
                core::StateBuilder::newState(SampledPosition)));
        }
    }

    // Setup a config.
    auto Config = std::make_unique<core::StateConfig>();

    static std::uniform_int_distribution<> MaxPlyDistribution(160 + 64,
                                                              512 + 128);
    static std::uniform_real_distribution<float> DrawRateDistribution(0.0f,
                                                                      1.0f);

    Config->MaxPly = (uint16_t)MaxPlyDistribution(MT);

    uint64_t R = MT() % 4;
    Config->Rule = core::EndingRule::ER_Declare27;
    if (R < 2) {
        Config->BlackDrawValue = 0.5f;
        Config->WhiteDrawValue = 0.5f;
    } else {
        Config->BlackDrawValue = DrawRateDistribution(MT);
        Config->WhiteDrawValue = 1.0f - Config->BlackDrawValue;
    }

    F->setConfig(std::move(Config));

    F->clearDidFullSearch();

    return SelfplayPhase::RootPreparation;
}

SelfplayPhase Worker::prepareRoot(Frame* F) const {
    // Update the search tree.
    F->getSearchTree()->updateRoot(*F->getState(), false);
    F->setRootPly(F->getState()->getPly());

    // Sample noises.
    // We don't care about actual the number of legal moves.
    // In stead, we prepare sufficient enough number of gumbel noises.
    for (std::size_t I = 0; I < F->getNoise().size(); ++I) {
        F->getNoise().at(I) = sampleNoise(F);
    }

    if (!F->isGumbel()) {
        const double Sum =
            std::accumulate(F->getNoise().begin(), F->getNoise().end(), 0.0);
        for (std::size_t I = 0; I < F->getNoise().size(); ++I) {
            F->getNoise().at(I) /= Sum;
        }
    }

    std::uniform_real_distribution<double> Distribution(0.0, 1.0);
    const double R = Distribution(MT);
    const auto Moves = core::MoveGenerator::generateLegalMoves(*F->getState());
    if (Moves.size() == 1 || R > MyFullSearchRatio) {
        if (F->isGumbel()) {
            // NumPlayouts == NumSamplingMoves.
            F->setNumPlayouts(MyNumSamplingMoves);
            F->setNumSamplingMoves(MyNumSamplingMoves);
        } else {
            if (Moves.size() == 1) {
                F->setNumPlayouts(1);
            } else {
                F->setNumPlayouts(std::max((uint64_t)1, MyNumPlayouts / 4));
            }
        }
        F->pushDidFullSearch(false);
    } else {
        // Full search.
        F->setNumPlayouts(MyNumPlayouts);
        if (F->isGumbel()) {
            F->setNumSamplingMoves(MyNumSamplingMoves);
        }
        F->pushDidFullSearch(true);
    }

    if (F->isGumbel()) {
        const uint64_t InitialSequentialHalvingPlayouts = (uint64_t)(std::floor(
            (double)F->getNumPlayouts() /
            (std::log2(F->getNumSamplingMoves()) * F->getNumSamplingMoves())));

        F->setSequentialHalvingPlayouts(
            std::max((uint64_t)1, InitialSequentialHalvingPlayouts));
        F->setSequentialHalvingCount(1);
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

        if (Node->getRepetitionStatus() !=
            core::RepetitionStatus::NoRepetition) {
            break;
        }

        if (F->getState()->getPly() >= F->getStateConfig()->MaxPly) {
            break;
        }

        mcts::Edge* E =
            pickUpEdgeToExplore(F, F->getState()->getSideToMove(), Node, Depth);
        assert(E != nullptr);

        F->getState()->doMove(F->getState()->getMove32FromMove16(E->getMove()));

        if (E->getTarget() == nullptr) {
            mcts::Pointer<mcts::Node> NewNode;
            NewNode.malloc(NA, Node);
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
    // This node has been already evaluated.
    if (F->getNodeToEvaluate()->getVisitsAndVirtualLoss() > 0) {
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        return SelfplayPhase::Backpropagation;
    }

    const auto RS = F->getState()->getRepetitionStatus(true);

    // Repetition.
    if (RS == core::RepetitionStatus::WinRepetition) {
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        F->setEvaluation<true>(nullptr, 1.0f, 0.0f);
        F->getNodeToEvaluate()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::LossRepetition) {
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        F->setEvaluation<true>(nullptr, 0.0f, 0.0f);
        F->getNodeToEvaluate()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    } else if (RS == core::RepetitionStatus::Repetition) {
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        const float DrawValue = F->getState()->getSideToMove() == core::Black
                                    ? F->getStateConfig()->BlackDrawValue
                                    : F->getStateConfig()->WhiteDrawValue;
        F->setEvaluation<true>(nullptr, DrawValue, 1.0f);
        F->getNodeToEvaluate()->setRepetitionStatus(RS);
        return SelfplayPhase::Backpropagation;
    }

    // Declaration.
    if (F->getStateConfig()->Rule == core::EndingRule::ER_Declare27) {
        if (F->getState()->canDeclare()) {
#ifndef NDEBUG
            if (F->getNodeToEvaluate() == F->getSearchTree()->getRoot()) {
                std::cerr << "State: "
                          << nshogi::io::sfen::stateToSfen(*F->getState())
                          << std::endl;
                std::cerr << "Error: The root node cannot be a declaration."
                          << std::endl;
                std::abort();
            }
#endif
            assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
            F->setEvaluation<true>(nullptr, 1.0f, 0.0f);
            return SelfplayPhase::Backpropagation;
        }
    }

    const auto LegalMoves =
        nshogi::core::MoveGenerator::generateLegalMoves(*F->getState());

    // Checkmate.
    if (LegalMoves.size() == 0) {
        if (F->getState()->getPly(false) > 0) {
            // Check if the last move is a checkmate by dropping a pawn.
            const auto LastMove = F->getState()->getLastMove();
            if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
                F->setEvaluation<true>(nullptr, 1.0f, 0.0f);
                return SelfplayPhase::Backpropagation;
            }
        }

        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        F->setEvaluation<true>(nullptr, 0.0f, 0.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Max ply.
    if (F->getState()->getPly() >= F->getStateConfig()->MaxPly) {
        const float DrawValue = F->getState()->getSideToMove() == core::Black
                                    ? F->getStateConfig()->BlackDrawValue
                                    : F->getStateConfig()->WhiteDrawValue;
        assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
        F->setEvaluation<true>(nullptr, DrawValue, 1.0f);
        return SelfplayPhase::Backpropagation;
    }

    // Checkmate by search.
    if (F->getState()->getPly() > F->getRootPly()) {
        if (isCheckmated(F)) {
            F->setEvaluation<true>(nullptr, 0.0f, 0.0f);
            assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
            return SelfplayPhase::Backpropagation;
        } else if (!solver::dfs::solve(F->getState(), 3).isNone()) {
            F->setEvaluation<true>(nullptr, 1.0f, 0.0f);
            assert(F->getNodeToEvaluate() != F->getSearchTree()->getRoot());
            return SelfplayPhase::Backpropagation;
        }
    }

    const int16_t NumEdges = F->getNodeToEvaluate()->expand(LegalMoves, EA);
    if (NumEdges < 0) {
        std::cerr << "Error: Failed to allocate memory for edges." << std::endl;
        std::abort();
    }

    // Check evaluation cache.
    assert(EvalCache != nullptr);
    mcts::EvalCache::EvalInfo EvalInfo;
    if (EvalCache->load(*F->getState(), &EvalInfo)) {
        if (EvalInfo.NumMoves == LegalMoves.size()) {
            F->setEvaluation<true>(EvalInfo.Policy, EvalInfo.WinRate,
                                   EvalInfo.DrawRate);
            SInfo->incrementCacheHit();
            return SelfplayPhase::Backpropagation;
        }
    }
    SInfo->incrementCacheMiss();

    return SelfplayPhase::Evaluation;
}

SelfplayPhase Worker::backpropagate(Frame* F) const {
    // Backpropagate win rate and draw rate.
    F->getNodeToEvaluate()->updateAncestors<false>(
        F->getNodeToEvaluate()->getWinRatePredicted(),
        F->getNodeToEvaluate()->getDrawRatePredicted());

    while (F->getState()->getPly() > F->getRootPly()) {
        F->getState()->undoMove();
    }

#ifndef NDEBUG
    if (F->getSearchTree()->getRoot()->getNumChildren() == 0) {
        std::cerr << "State: " << nshogi::io::sfen::stateToSfen(*F->getState())
                  << std::endl;
        std::cerr << "Moves: ";
        const auto Moves =
            core::MoveGenerator::generateLegalMoves(*F->getState());
        for (const core::Move32 Move : Moves) {
            std::cerr << nshogi::io::sfen::move32ToSfen(Move) << ", ";
        }
        std::cerr << std::endl;

        std::cerr << "Error: No legal moves at the root node." << std::endl;
        std::abort();
    }
#endif
    return SelfplayPhase::SequentialHalving;
}

SelfplayPhase Worker::sequentialHalving(Frame* F) const {
    // Since we have identified the number of legal moves at the root is one,
    // we don't need to search further more and go to the transition phase.
    if (F->getSearchTree()->getRoot()->getNumChildren() == 1) {
        return SelfplayPhase::Transition;
    }

    if (!F->isGumbel()) {
        if (F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss() >=
            F->getNumPlayouts()) {
            return SelfplayPhase::Transition;
        }

        // Nothing to do in AlphaZero style.
        return SelfplayPhase::LeafSelection;
    }

    // If the node is root node, extract top m moves sorted by
    // gumbel noise and policy.
    if (F->getNodeToEvaluate() == F->getSearchTree()->getRoot()) {
        assert(F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss() == 1);
        sampleTopMMoves(F);
    } else {
        assert(F->getSearchTree()->getRoot()->getNumChildren() ==
               F->getIsTarget().size());
        uint64_t MinN = std::numeric_limits<uint64_t>::max();
        for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
            if (!F->getIsTarget().at(I)) {
                continue;
            }

            mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];
            mcts::Node* Child = Edge->getTarget();

            if (Child == nullptr) {
                MinN = 0;
                break;
            }

            MinN = std::min(MinN, Child->getVisitsAndVirtualLoss());
        }

        if (MinN >= F->getSequentialHalvingPlayouts()) {
            // The number of simulation exceeds the requirement, stop searching
            // at this node and proceed to a next state.
            if (F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss() >=
                F->getNumPlayouts() + 1) {
                // Adding 1 because it consumes one simulation to evaluate
                // the root node itself.
                return SelfplayPhase::Transition;
            }

            const uint16_t NumValidChilds = executeSequentialHalving(F);
            assert(NumValidChilds >= 2);
            const bool ExtraVisitsLeft =
                updateSequentialHalvingSchedule(F, NumValidChilds);
            if (!ExtraVisitsLeft) {
                return SelfplayPhase::Transition;
            }
        }
    }

    return SelfplayPhase::LeafSelection;
}

SelfplayPhase Worker::judge(Frame* F) const {
    const auto RS = F->getState()->getRepetitionStatus(true);

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

    if (F->getStateConfig()->Rule == core::EndingRule::ER_Declare27 &&
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

    const auto CheckmateMove = Solver.solve(F->getState(), 100000, 0);

    if (!CheckmateMove.isNone()) {
        F->setWinner(F->getState()->getSideToMove());
        F->getState()->doMove(CheckmateMove);
        F->pushDidFullSearch(true);
        return SelfplayPhase::Save;
    }

    return SelfplayPhase::RootPreparation;
}

SelfplayPhase Worker::transition(Frame* F) const {
    while (F->getState()->getPly() > F->getRootPly()) {
        F->getState()->undoMove();
    }

    if (!F->isGumbel()) { // AlphaZero style.
        // Choose a next move proportionally to visit counts.
        // Currently disabled because self-play games start
        // from various initial positions.
        if (false) {
            std::vector<double> VisitCounts;
            VisitCounts.reserve(
                F->getSearchTree()->getRoot()->getNumChildren());
            for (std::size_t I = 0;
                 I < F->getSearchTree()->getRoot()->getNumChildren(); ++I) {
                mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];
                mcts::Node* Child = Edge->getTarget();

                if (Child == nullptr) {
                    VisitCounts.push_back(0.0);
                } else {
                    VisitCounts.push_back(
                        (double)Child->getVisitsAndVirtualLoss());
                }
            }

            std::discrete_distribution<std::size_t> Distribution(
                VisitCounts.begin(), VisitCounts.end());
            const std::size_t SelectedIndex = Distribution(MT);

            mcts::Edge* SelectedEdge =
                &F->getSearchTree()->getRoot()->getEdge()[SelectedIndex];
            F->getState()->doMove(
                F->getState()->getMove32FromMove16(SelectedEdge->getMove()));
        } else {
            // Choose the most visited move.
            uint64_t MaxVisits = 0;
            mcts::Edge* MaxEdge = nullptr;
            for (std::size_t I = 0;
                 I < F->getSearchTree()->getRoot()->getNumChildren(); ++I) {
                mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];
                mcts::Node* Child = Edge->getTarget();

                if (Child == nullptr) {
                    if (MaxEdge == nullptr) {
                        MaxEdge = Edge;
                    } else if (MaxEdge->getTarget() == nullptr &&
                               Edge->getProbability() >
                                   MaxEdge->getProbability()) {
                        MaxEdge = Edge;
                    }
                    continue;
                }

                if (Child->getVisitsAndVirtualLoss() > MaxVisits) {
                    MaxVisits = Child->getVisitsAndVirtualLoss();
                    MaxEdge = Edge;
                } else if (Child->getVisitsAndVirtualLoss() == MaxVisits) {
                    if (Edge->getProbability() > MaxEdge->getProbability()) {
                        MaxEdge = Edge;
                    }
                }
            }

            assert(MaxEdge != nullptr ||
                   F->getSearchTree()->getRoot()->getNumChildren() == 1);
            F->getState()->doMove(
                F->getState()->getMove32FromMove16(MaxEdge->getMove()));
        }
        return SelfplayPhase::Judging;
    }

    if (F->getSearchTree()->getRoot()->getNumChildren() == 1) {
        const mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[0];
        F->getState()->doMove(
            F->getState()->getMove32FromMove16(Edge->getMove()));
        return SelfplayPhase::Judging;
    }

    double ScoreMax = std::numeric_limits<double>::lowest();
    mcts::Edge* ScoreMaxEdge = nullptr;

    assert(F->getIsTarget().size() > 0);
    for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
        if (!F->getIsTarget().at(I)) {
            continue;
        }

        mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];
        mcts::Node* Child = Edge->getTarget();
        assert(Child != nullptr);

        const double Score =
            F->getNoise().at(I) + Edge->getProbability() +
            transformQ(
                computeWinRateOfChild(F, F->getState()->getSideToMove(), Child),
                Child->getVisitsAndVirtualLoss());
        // Note: (any) Child->Visits == \max_{Child} Child->Visits
        // holds becasue the children of the root node are visited equally.

        if (Score > ScoreMax) {
            ScoreMax = Score;
            ScoreMaxEdge = Edge;
        }
    }

    assert(ScoreMaxEdge != nullptr);
    F->getState()->doMove(
        F->getState()->getMove32FromMove16(ScoreMaxEdge->getMove()));
    return SelfplayPhase::Judging;
}

double Worker::sampleNoise(Frame* F) const {
    if (F->isGumbel()) {
        // Gumbel noise.
        std::uniform_real_distribution<double> Distribution(
            std::numeric_limits<double>::min(), 1.0);

        const double U = Distribution(MT);
        return -std::log(-std::log(U));
    } else {
        // Dirichlet noise.
        constexpr double Alpha = 0.15;
        std::gamma_distribution<double> Distribution(Alpha, 1.0);
        return Distribution(MT);
    }
}

double Worker::transformQ(double Q, uint64_t MaxN) const {
    constexpr double C_VISIT = 50.0;
    constexpr double C_SCALE = 1.0;

    return (C_VISIT + (double)MaxN) * C_SCALE * Q;
}

template <>
mcts::Edge* Worker::pickUpEdgeToExplore<true>(Frame* F, core::Color,
                                              mcts::Node* N) const {
    const uint16_t NumChildren = N->getNumChildren();
    for (std::size_t I = 0; I < NumChildren; ++I) {
        if (!F->getIsTarget().at(I)) {
            // This edge is disabled (i.e., this node was not
            // chosen in the first sampling of m children).
            continue;
        }

        mcts::Edge* Edge = &N->getEdge()[I];
        mcts::Node* Child = Edge->getTarget();

        if (Child == nullptr || (Child->getVisitsAndVirtualLoss() <
                                 F->getSequentialHalvingPlayouts())) {
            return Edge;
        }
    }

    assert(false);
    return nullptr;
}

template <>
mcts::Edge* Worker::pickUpEdgeToExplore<false>(Frame* F, core::Color SideToMove,
                                               mcts::Node* N) const {
    double ScoreMax = std::numeric_limits<double>::lowest();
    mcts::Edge* EdgeToExplore = nullptr;
    const uint16_t NumChildren = N->getNumChildren();

    const double C = 1.25 * std::sqrt((double)N->getVisitsAndVirtualLoss());

    for (std::size_t I = 0; I < NumChildren; ++I) {
        mcts::Edge* Edge = &N->getEdge()[I];
        mcts::Node* Child = Edge->getTarget();

        const double Score =
            (Child == nullptr)
                ? (C * Edge->getProbability())
                : (C * Edge->getProbability() /
                       ((double)(1 + Child->getVisitsAndVirtualLoss())) +
                   computeWinRateOfChild(F, SideToMove, Child));

        if (Score > ScoreMax) {
            ScoreMax = Score;
            EdgeToExplore = Edge;
        }
    }

    assert(EdgeToExplore != nullptr);
    return EdgeToExplore;
}

mcts::Edge* Worker::pickUpEdgeToExplore(Frame* F, core::Color SideToMove,
                                        mcts::Node* N, uint8_t Depth) const {
    if (F->isGumbel()) {
        return (Depth == 0) ? pickUpEdgeToExplore<true>(F, SideToMove, N)
                            : pickUpEdgeToExplore<false>(F, SideToMove, N);
    } else {
        return pickUpEdgeToExplore<false>(F, SideToMove, N);
    }
}

double Worker::computeWinRate(Frame* F, core::Color SideToMove,
                              mcts::Node* Node) const {
    const uint64_t Visits = Node->getVisitsAndVirtualLoss();
    assert(Visits > 0);

    const double WinRateAccumulated = Node->getWinRateAccumulated();
    const double DrawRateAccumulated = Node->getDrawRateAccumulated();

    const double WinRate = WinRateAccumulated / (double)Visits;
    const double DrawRate = DrawRateAccumulated / (double)Visits;

    const double DrawValue = (SideToMove == core::Black)
                                 ? F->getStateConfig()->BlackDrawValue
                                 : F->getStateConfig()->WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

double Worker::computeWinRateOfChild(Frame* F, core::Color SideToMove,
                                     mcts::Node* Child) const {
    const uint64_t ChildVisits = Child->getVisitsAndVirtualLoss();
    assert(ChildVisits > 0);

    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAcuumulated = Child->getDrawRateAccumulated();

    const double WinRate =
        ((double)ChildVisits - ChildWinRateAccumulated) / (double)ChildVisits;
    const double DrawRate = ChildDrawRateAcuumulated / (double)ChildVisits;

    const double DrawValue = (SideToMove == core::Black)
                                 ? F->getStateConfig()->BlackDrawValue
                                 : F->getStateConfig()->WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

bool Worker::isCheckmated(Frame* F) const {
    if (!F->getState()->isInCheck()) {
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

void Worker::sampleTopMMoves(Frame* F) const {
    assert(F->isGumbel());

    F->getIsTarget().clear();
    F->getIsTarget().resize(F->getSearchTree()->getRoot()->getNumChildren());

    // If the number of sampling moves is larger than or equal to
    // the number of the legal moves at the root node, we don't
    // need to sample the moves.
    if (F->getNumSamplingMoves() >= F->getIsTarget().size()) {
        std::fill(F->getIsTarget().begin(), F->getIsTarget().end(), true);
        return;
    }

    std::pair<double, std::size_t> ScoreWithIndex[600];
    for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
        mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];

        ScoreWithIndex[I].first = F->getNoise().at(I) + Edge->getProbability();
        ScoreWithIndex[I].second = I;
    }

    // Set top m moves to explore by gumbel noise and policy.
    const std::size_t NumSort = (std::size_t)F->getNumSamplingMoves();

    std::partial_sort(ScoreWithIndex, ScoreWithIndex + (long)NumSort,
                      ScoreWithIndex + (long)F->getIsTarget().size(),
                      [](const std::pair<double, std::size_t>& Elem1,
                         const std::pair<double, std::size_t>& Elem2) {
                          return Elem1.first > Elem2.first;
                      });
    std::fill(F->getIsTarget().begin(), F->getIsTarget().end(), false);
    for (std::size_t I = 0; I < NumSort; ++I) {
        F->getIsTarget().at(ScoreWithIndex[I].second) = true;
    }
}

uint16_t Worker::executeSequentialHalving(Frame* F) const {
    assert(F->isGumbel());
    assert(F->getSearchTree()->getRoot()->getNumChildren() ==
           F->getIsTarget().size());
    assert(F->getState()->getPly() >= F->getRootPly());

    std::pair<double, std::size_t> ScoreWithIndex[600];
    for (std::size_t I = 0; I < F->getIsTarget().size(); ++I) {
        if (!F->getIsTarget().at(I)) {
            ScoreWithIndex[I].first = std::numeric_limits<double>::lowest();
            continue;
        }

        mcts::Edge* Edge = &F->getSearchTree()->getRoot()->getEdge()[I];
        mcts::Node* Child = Edge->getTarget();

        assert(Child != nullptr);
        ScoreWithIndex[I].first =
            F->getNoise().at(I) + Edge->getProbability() +
            transformQ(
                computeWinRateOfChild(F, F->getState()->getSideToMove(), Child),
                Child->getVisitsAndVirtualLoss());
        // Note: (any) Child->Visits == \max_{Child} Child->Visits
        // holds becasue the children of the root node are visited equally.
        ScoreWithIndex[I].second = I;
    }

    std::size_t NumSort = std::min((std::size_t)F->getNumSamplingMoves(),
                                   F->getIsTarget().size());
    assert(NumSort > 1);
    assert(F->getSequentialHalvingCount() > 0);
    NumSort =
        std::max((uint64_t)2, (NumSort + 1) >> F->getSequentialHalvingCount());

    // Gather top moves along policy + gumbel noise.
    std::partial_sort(ScoreWithIndex, ScoreWithIndex + (long)NumSort,
                      ScoreWithIndex + (long)F->getIsTarget().size(),
                      [](const std::pair<double, std::size_t>& Elem1,
                         const std::pair<double, std::size_t>& Elem2) {
                          return Elem1.first > Elem2.first;
                      });
    std::fill(F->getIsTarget().begin(), F->getIsTarget().end(), false);
    for (std::size_t I = 0; I < NumSort; ++I) {
        F->getIsTarget().at(ScoreWithIndex[I].second) = true;
    }

    return (uint16_t)NumSort;
}

bool Worker::updateSequentialHalvingSchedule(Frame* F,
                                             uint16_t NumValidChilds) const {
    assert(NumValidChilds >= 2);

    const uint16_t MD =
        std::max((uint16_t)1, (uint16_t)(F->getNumSamplingMoves() >>
                                         F->getSequentialHalvingCount()));
    const double D = std::log2((double)F->getNumSamplingMoves()) * (double)MD;
    const uint64_t ExtraVisits =
        (uint64_t)(std::floor((double)F->getNumPlayouts() / D));

    if (ExtraVisits == 0) {
        return false;
    }

    if (NumValidChilds <= 2) {
        assert(NumValidChilds == 2);

        // Use all left simulation budgets to identify the best move.
        assert(F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss() > 0);
        assert(F->getNumPlayouts() + 1 >
               F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss());
        const uint64_t LeftPlayouts =
            F->getNumPlayouts() + 1 -
            F->getSearchTree()->getRoot()->getVisitsAndVirtualLoss();
        F->setSequentialHalvingPlayouts(F->getSequentialHalvingPlayouts() +
                                        (LeftPlayouts + 1) / 2);
        F->setSequentialHalvingCount(F->getSequentialHalvingCount() + 1);
    } else {
        F->setSequentialHalvingPlayouts(F->getSequentialHalvingPlayouts() +
                                        ExtraVisits);
        F->setSequentialHalvingCount(F->getSequentialHalvingCount() + 1);
    }

    return true;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
