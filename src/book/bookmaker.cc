//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookmaker.h"

#include <cassert>
#include <cmath>
#include <chrono>
#include <thread>

#include <nshogi/core/state.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>


#include <nshogi/solver/mate1ply.h>

namespace nshogi {
namespace engine {
namespace book {

BookMaker::BookMaker()
    : Solver(2048) {

}

void BookMaker::start(const core::State& RootState, uint64_t NumSimulations) {
    // TODO: root state must be the actual root state.

    if (Nodes.empty()) {
        prepareRoot(RootState);
    }

    for (uint64_t Simulation = 0; Simulation < NumSimulations; ++Simulation) {
        core::State State = RootState.clone();

        // Step1: Select one leaf node.
        NodeIndex LeafNode = collectOneLeaf(&State);

        // Step2: Expand the leaf node and evaluate it.
        if (Nodes[LeafNode].VisitCount == 0) {
            expandAndEvaluate(&State, LeafNode);
        }

        // Step3: Backpropagate the evaluation result.
        backpropagate(LeafNode, Nodes[LeafNode].WinRateRaw, Nodes[LeafNode].DrawRateRaw);

        if (Simulation % 100 == 0) {
            debugOutput();
        }
    }
}

void BookMaker::prepareRoot(const core::State& RootState) {
    // Nodes[0] is the null node.
    Nodes.emplace_back(NI_Null);

    // Nodes[1] is the root node.
    Nodes.emplace_back(NI_Root);

    NodeIndices[io::sfen::positionToSfen(RootState.getPosition())] = NI_Root;
}

NodeIndex BookMaker::collectOneLeaf(core::State* State) {
    NodeIndex CurrentNode = NI_Root;

    while (true) {
        if (Nodes[CurrentNode].Children.empty()) {
            return Nodes[CurrentNode].Index;
        }

        // Register child if it has been registered somewhere.
        for (std::size_t I = 0; I < Nodes[CurrentNode].Moves.size(); ++I) {
            if (Nodes[CurrentNode].Children[I] == NI_Null) {
                const auto Move = Nodes[CurrentNode].Moves[I];
                State->doMove(Move);

                if (State->getRepetitionStatus() == core::RepetitionStatus::NoRepetition) {
                    const auto It =
                        NodeIndices.find(io::sfen::positionToSfen(State->getPosition()));
                    if (It != NodeIndices.end()) {
                        Nodes[CurrentNode].Children[I] = It->second;
                        Nodes[It->second].Parents.push_back(Nodes[CurrentNode].Index);
                    }
                }

                State->undoMove();
            }
        }

        auto [BestChildIndex, BestMove] = computeUCBMaxChild(State, CurrentNode);

        CurrentNode = BestChildIndex;
        State->doMove(BestMove);
    }
}

void BookMaker::expandAndEvaluate(core::State* State, NodeIndex N) {
    const auto RS = State->getRepetitionStatus();
    if (RS == core::RepetitionStatus::WinRepetition ||
        RS == core::RepetitionStatus::SuperiorRepetition) {
        Nodes[N].WinRateRaw = 1.0f;
        Nodes[N].DrawRateRaw = 0.0f;
        return;
    }

    if (RS == core::RepetitionStatus::LossRepetition ||
        RS == core::RepetitionStatus::InferiorRepetition) {
        Nodes[N].WinRateRaw = 0.0f;
        Nodes[N].DrawRateRaw = 0.0f;
        return;
    }

    if (RS == core::RepetitionStatus::Repetition) {
        Nodes[N].WinRateRaw = 0.0f;
        Nodes[N].DrawRateRaw = 1.0f;
        return;
    }

    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);

    if (Moves.size() == 0) {
        if (State->getPly() > 0) {
            const auto LastMove = State->getLastMove();
            if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                Nodes[N].WinRateRaw = 1.0f;
                Nodes[N].DrawRateRaw = 0.0f;
                return;
            }
        }

        Nodes[N].WinRateRaw = 0.0f;
        Nodes[N].DrawRateRaw = 0.0f;
        return;
    }

    // Checkmate search.
    // TODO
    // const auto CheckmateMove = Solver.solve(State);
    // if (!CheckmateMove.isNone()) {
    //     N->WinRateRaw = 1.0f;
    //     N->DrawRateRaw = 0.0f;
    //     return;
    // }
    const auto CheckmateMove = solver::mate1ply::solve(*State);
    if (!CheckmateMove.isNone()) {
        Nodes[N].WinRateRaw = 1.0f;
        Nodes[N].DrawRateRaw = 0.0f;
        return;
    }

    for (const auto& Move : Moves) {
        Nodes[N].Moves.push_back(Move);
    }

    evaluate(N);
}

void BookMaker::evaluate(NodeIndex N) {
    // TODO: actual evaluation.
    // Dummy evaluation: uniform distribution.
    for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
        Nodes[N].Policies.push_back(1.0f / (float)Nodes[N].Moves.size());
        Nodes[N].Children.push_back(NI_Null);
    }
    Nodes[N].WinRateRaw = 0.5f;
    Nodes[N].DrawRateRaw = 0.0f;
    // std::this_thread::sleep_for(std::chrono::seconds(1));
}

void BookMaker::backpropagate(NodeIndex N, float WinRate, float DrawRate) {
    Nodes[N].VisitCount += 1;
    Nodes[N].WinRateAccumulated += WinRate;
    Nodes[N].DrawRateAccumulated += DrawRate;

    for (const auto& ParentIndex : Nodes[N].Parents) {
        backpropagate(ParentIndex, 1.0f - WinRate, DrawRate);
    }
}

std::pair<NodeIndex, core::Move32> BookMaker::computeUCBMaxChild(core::State* State, NodeIndex N) {
    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;

    std::size_t BestIndex = 0;
    core::Move32 BestMove = core::Move32::MoveNone();
    double UCBMaxValue = -1.0;

    const double Const =
        (std::log((double)(Nodes[N].VisitCount + CBase) / (double)CBase) + CInit) *
        std::sqrt((double)Nodes[N].VisitCount);

    for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
        double Q = 0.0;
        uint64_t ChildVisitCount = 0;

        if (Nodes[N].Children[I] != NI_Null) {
            Node* Child = &Nodes[Nodes[N].Children[I]];
            // Reverse win rate because the side to move is changed.
            Q = 1.0 - (Child->WinRateAccumulated / (double)Child->VisitCount);
            ChildVisitCount = Child->VisitCount;
        }

        const double UCBValue = Q + Const * Nodes[N].Policies[I] / (double)(1 + ChildVisitCount);

        if (UCBValue > UCBMaxValue) {
            UCBMaxValue = UCBValue;
            BestIndex = I;
            BestMove = Nodes[N].Moves[I];
        }
    }

    // Create child node if it does not exist.
    if (Nodes[N].Children[BestIndex] == NI_Null) {
        Nodes.emplace_back((NodeIndex)(Nodes.size()), Nodes[N].Index);
        Nodes[N].Children[BestIndex] = (NodeIndex)(Nodes.size() - 1);
    }

    assert(!BestMove.isNone());
    return { Nodes[N].Children[BestIndex], BestMove };
}

void BookMaker::debugOutput() const {
    const Node* Root = &Nodes[NI_Root];

    std::cout << "[Root]" << std::endl;
    std::cout << "    - Visits: " << Root->VisitCount << std::endl;
    std::cout << "    - WinRate: " << (Root->WinRateAccumulated / (double)Root->VisitCount) << std::endl;
    std::cout << "    - DrawRate: " << (Root->DrawRateAccumulated / (double)Root->VisitCount) << std::endl;
    std::cout << "    - PV: ";
    const auto PV = currentPV(NI_Root);
    for (const auto& Move : PV) {
        std::cout << io::sfen::move32ToSfen(Move) << " ";
    }
    std::cout << std::endl;
}

std::vector<core::Move32> BookMaker::currentPV(NodeIndex N) const {
    std::vector<core::Move32> PV;

    const Node* CurrentNode = &Nodes[N];

    while (true) {
        if (CurrentNode->Children.empty()) {
            break;
        }

        // Find the most visited child.
        std::size_t BestIndex = 0;
        uint64_t MaxVisits = 0;

        for (std::size_t I = 0; I < CurrentNode->Moves.size(); ++I) {
            if (CurrentNode->Children[I] != NI_Null) {
                const Node* Child = &Nodes[CurrentNode->Children[I]];
                if (Child->VisitCount > MaxVisits) {
                    MaxVisits = Child->VisitCount;
                    BestIndex = I;
                }
            }
        }

        if (MaxVisits == 0) {
            break;
        }

        PV.push_back(CurrentNode->Moves[BestIndex]);
        CurrentNode = &Nodes[CurrentNode->Children[BestIndex]];
    }

    return PV;
}

} // namespace book
} // namespace engine
} // namespace nshogi
