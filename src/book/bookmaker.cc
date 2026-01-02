//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookmaker.h"
#include "../limit.h"
#include "../protocol/usilogger.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <unordered_set>

#include <nshogi/core/state.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>
#include <nshogi/solver/mate1ply.h>

namespace nshogi {
namespace engine {
namespace book {

BookMaker::BookMaker()
    : Solver(2048) {

    // Load book/ckpt_latest.index and book/ckpt_latest.data if exists.
    {
        std::ifstream IndexIfs("book_ckpt/ckpt_latest.index", std::ios::binary);
        std::ifstream DataIfs("book_ckpt/ckpt_latest.data", std::ios::binary);
        if (IndexIfs && DataIfs) {
            io::book::load(this, IndexIfs, DataIfs);
            std::cout << "[BookMaker] Loaded checkpoint from book_ckpt/ckpt_latest." << std::endl;
        }
    }
}

void BookMaker::start(const core::State& RootState, uint64_t NumSimulations) {
    // TODO: resume with a non-root state.

    prepareMCTSManager();

    if (Nodes.empty()) {
        prepareRoot(RootState);
    }

    for (uint64_t Simulation = 0; Simulation < NumSimulations; ++Simulation) {
        core::State State = RootState.clone();

        // Step1: Select one leaf node.
        auto [Trajectory, LeafWinRate, LeafDrawRate] = collectOneLeaf(&State);
        const auto LeafNodeIndex = Trajectory.back().first;
        Trajectory.pop_back();

        // Step2: Expand the leaf node and evaluate it.
        if (Nodes[LeafNodeIndex].visitCount() == 0) {
            expandAndEvaluate(&State, LeafNodeIndex);
            LeafWinRate = Nodes[LeafNodeIndex].WinRateRaw;
            LeafDrawRate = Nodes[LeafNodeIndex].DrawRateRaw;
        }

        // Step3: Backpropagate the evaluation result.
        backpropagate(Trajectory, LeafWinRate, LeafDrawRate);

        if (Simulation % 1 == 0) {
            debugOutput();
        }

        if (Simulation % 500 == 0) {
            {
                const std::string IndexOutputPath = "book_ckpt/ckpt_" + std::to_string(Simulation) + ".index";
                const std::string DataOutputPath = "book_ckpt/ckpt_" + std::to_string(Simulation) + ".data";

                std::ofstream IndexOfs(IndexOutputPath, std::ios::binary);
                std::ofstream DataOfs(DataOutputPath, std::ios::binary);
                io::book::save(*this, IndexOfs, DataOfs);
            }

            {
                const std::string IndexOutputPath = "book_ckpt/ckpt_latest.index";
                const std::string DataOutputPath = "book_ckpt/ckpt_latest.data";

                std::ofstream IndexOfs(IndexOutputPath, std::ios::binary);
                std::ofstream DataOfs(DataOutputPath, std::ios::binary);
                io::book::save(*this, IndexOfs, DataOfs);
            }

            std::cout << "[BookMaker] Checkpointed at simulation " << Simulation << "." << std::endl;
        }
    }
}

void BookMaker::prepareMCTSManager() {
    MCTSManager.reset();

    CManager.setIsPonderingEnabled(false);
    CManager.setMinimumThinkinTimeMilliSeconds(2 * 1000);
    CManager.setMaximumThinkinTimeMilliSeconds(2 * 1000);
    CManager.setNumCheckmateSearchThreads(8);
    CManager.setBookEnabled(false);
    CManager.setIsThoughtLogEnabled(true);
    CManager.setPrintStatistics(false);

    std::shared_ptr<logger::Logger> Logger =
        std::make_shared<protocol::usi::USILogger>();
    MCTSManager = std::make_unique<mcts::Manager>(CManager.getContext(), std::move(Logger));

    StateConfig.Rule = core::ER_Declare27;
    StateConfig.MaxPly = 512;
    StateConfig.BlackDrawValue = 0.5;
    StateConfig.WhiteDrawValue = 0.5;
}

void BookMaker::prepareRoot(const core::State& RootState) {
    // Nodes[0] is the null node.
    Nodes.emplace_back(NI_Null);

    // Nodes[1] is the root node.
    Nodes.emplace_back(NI_Root);

    NodeIndices[nshogi::io::sfen::positionToSfen(RootState.getPosition())] = NI_Root;
}

std::tuple<std::vector<std::pair<NodeIndex, std::size_t>>, float, float> BookMaker::collectOneLeaf(core::State* State) {
    std::vector<std::pair<NodeIndex, std::size_t>> Trajectory;

    NodeIndex CurrentNode = NI_Root;

    while (true) {
        if (Nodes[CurrentNode].Children.empty()) {
            Trajectory.push_back({ CurrentNode, 0 });
            return { std::move(Trajectory), 0.0f, 0.0f };
        }

        const auto RS = State->getRepetitionStatus();
        if (RS == core::RepetitionStatus::WinRepetition ||
            RS == core::RepetitionStatus::SuperiorRepetition) {
            Trajectory.push_back({ CurrentNode, 0 });
            return { std::move(Trajectory), 1.0f, 0.0f };
        }
        if (RS == core::RepetitionStatus::LossRepetition ||
            RS == core::RepetitionStatus::InferiorRepetition) {
            Trajectory.push_back({ CurrentNode, 0 });
            return { std::move(Trajectory), 0.0f, 0.0f };
        }
        if (RS == core::RepetitionStatus::Repetition) {
            Trajectory.push_back({ CurrentNode, 0 });
            return { std::move(Trajectory), 0.0f, 1.0f };
        }

        // Register child if it has been registered somewhere.
        for (std::size_t I = 0; I < Nodes[CurrentNode].Moves.size(); ++I) {
            if (Nodes[CurrentNode].Children[I] == NI_Null) {
                const auto Move = Nodes[CurrentNode].Moves[I];
                State->doMove(Move);

                if (State->getRepetitionStatus() == core::RepetitionStatus::NoRepetition) {
                    const auto It =
                        NodeIndices.find(nshogi::io::sfen::positionToSfen(State->getPosition()));
                    if (It != NodeIndices.end()) {
                        Nodes[CurrentNode].Children[I] = It->second;
                    }
                }

                State->undoMove();
            }
        }

        auto [BestChildIndex, BestMove] = computeUCBMaxChild(State, CurrentNode);
        Trajectory.push_back({ CurrentNode, BestChildIndex });

        CurrentNode = Nodes[CurrentNode].Children[BestChildIndex];
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
    const auto CheckmateMove = Solver.solve(State);
    if (!CheckmateMove.isNone()) {
        Nodes[N].WinRateRaw = 1.0f;
        Nodes[N].DrawRateRaw = 0.0f;
        return;
    }

    for (const auto& Move : Moves) {
        Nodes[N].Moves.push_back(Move);
    }

    evaluate(State, N);
}

void BookMaker::evaluate(const core::State* State, NodeIndex N) {
    std::mutex Mutex;
    std::condition_variable CV;
    bool ThinkingDone = false;

    std::vector<std::pair<core::Move16, float>> Policy;
    float WinRate = 0.0;
    float DrawRate = 0.0;

    auto Callback = [&](core::Move32, std::unique_ptr<mcts::ThoughtLog> ThoughtLog) {
        WinRate = (float)ThoughtLog->WinRate;
        DrawRate = (float)ThoughtLog->DrawRate;

        uint64_t TotalVisitCounts = 0;
        for (const auto& VC : ThoughtLog->VisitCounts) {
            TotalVisitCounts += VC.second;
        }
        for (const auto& VC : ThoughtLog->VisitCounts) {
            const float P =  (float)VC.second / (float)TotalVisitCounts;
            Policy.push_back({ VC.first, P });
        }

        std::lock_guard<std::mutex> Lock(Mutex);
        ThinkingDone = true;
        CV.notify_one();
    };

    MCTSManager->thinkNextMove(
        *State,
        StateConfig,
        engine::NoLimit,
        Callback
    );

    { // Wait until evaluation is done.
        std::unique_lock<std::mutex> Lock(Mutex);
        CV.wait(Lock, [&ThinkingDone]() { return ThinkingDone; });
    }

    Nodes[N].PolicyRaw.resize(Nodes[N].Moves.size(), 0.0f);
    Nodes[N].VisitCounts.resize(Nodes[N].Moves.size(), 0);
    Nodes[N].WinRateAccumulateds.resize(Nodes[N].Moves.size(), 0.0);
    Nodes[N].DrawRateAccumulateds.resize(Nodes[N].Moves.size(), 0.0);
    Nodes[N].Children.resize(Nodes[N].Moves.size(), NI_Null);

    for (const auto& P : Policy) {
        for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
            if (core::Move16(Nodes[N].Moves[I]) == P.first) {
                Nodes[N].PolicyRaw[I] = P.second;
                break;
            }
        }
    }

    Nodes[N].WinRateRaw = (float)WinRate;
    Nodes[N].DrawRateRaw = (float)DrawRate;
}

void BookMaker::backpropagate(const std::vector<std::pair<NodeIndex, std::size_t>>& N, float WinRate, float DrawRate) {
    for (auto It = N.rbegin(); It != N.rend(); ++It) {
        // Flip the win rate for the opponent.
        WinRate = 1.0f - WinRate;

        const NodeIndex NodeIdx = It->first;
        const std::size_t MoveIdx = It->second;

        assert(MoveIdx < Nodes[NodeIdx].VisitCounts.size());

        Nodes[NodeIdx].VisitCounts[MoveIdx] += 1;
        Nodes[NodeIdx].WinRateAccumulateds[MoveIdx] += WinRate;
        Nodes[NodeIdx].DrawRateAccumulateds[MoveIdx] += DrawRate;
    }
}

std::pair<std::size_t, core::Move32> BookMaker::computeUCBMaxChild(core::State* State, NodeIndex N) {
    // TODO: draw value.

    // TODO based on the current win rate.

    const uint64_t ThisVisitCount = Nodes[N].visitCount() + 1;

    double WinRateMax = 0.0;
    for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
        if (Nodes[N].VisitCounts[I] == 0) {
            continue;
        }
        const double WR = Nodes[N].WinRateAccumulateds[I] / (double)Nodes[N].VisitCounts[I];
        if (WR > WinRateMax) {
            WinRateMax = WR;
        }
    }
    const double Const = std::sqrt((double)ThisVisitCount) *
        (WinRateMax > 0.52 ? 0.9 : 1.5);
    // static constexpr int32_t CBase = 19652;
    // static constexpr double CInit = 1.25;
    // const double Const =
    //     (std::log((double)(ThisVisitCount + CBase) / (double)CBase) + CInit) *
    //     std::sqrt((double)ThisVisitCount);

    std::size_t BestIndex = 0;
    core::Move32 BestMove = core::Move32::MoveNone();
    double UCBMaxValue = -1.0;

    for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
        const double Q = (Nodes[N].VisitCounts[I] == 0)
                             ? 0.0
                             : Nodes[N].WinRateAccumulateds[I] / (double)Nodes[N].VisitCounts[I];
        const double U = Const * Nodes[N].PolicyRaw[I] / (double)(1 + Nodes[N].VisitCounts[I]);
        const double UCBValue = Q + U;

        if (UCBValue > UCBMaxValue) {
            UCBMaxValue = UCBValue;
            BestIndex = I;
            BestMove = Nodes[N].Moves[I];
        }
    }

    // Create child node if it does not exist.
    if (Nodes[N].Children[BestIndex] == NI_Null) {
        const auto Move = Nodes[N].Moves[BestIndex];
        State->doMove(Move);

        Nodes.emplace_back((NodeIndex)(Nodes.size()));
        Nodes[N].Children[BestIndex] = (NodeIndex)(Nodes.size() - 1);

        // If the position has not been registered yet, register it.
        const auto Key = nshogi::io::sfen::positionToSfen(State->getPosition());
        if (auto It = NodeIndices.find(Key); It == NodeIndices.end()) {
            NodeIndices[Key] = Nodes[N].Children[BestIndex];
        }

        State->undoMove();
    }

    assert(!BestMove.isNone());
    return { BestIndex, BestMove };
}

void BookMaker::debugOutput() const {
    const Node* Root = &Nodes[NI_Root];

    std::cout << "====================" << std::endl;
    std::cout << "[Stats]" << std::endl;
    std::cout << "    - Total Nodes: " << Nodes.size() << std::endl;
    std::cout << "    - Index Size: " << NodeIndices.size() << std::endl;
    std::cout << "[Root]" << std::endl;
    std::cout << "    - Visits: " << Root->visitCount() + 1 << std::endl;

    std::cout << "    - WinRate (max): ";
    double MaxWinRate = 0.0;
    double DrawRate = 0.0;
    for (std::size_t I = 0; I < Root->Moves.size(); ++I) {
        if (Root->VisitCounts[I] == 0) {
            continue;
        }
        const double WR = Root->WinRateAccumulateds[I] / (double)Root->VisitCounts[I];
        if (WR > MaxWinRate) {
            MaxWinRate = WR;
            DrawRate = Root->DrawRateAccumulateds[I] / (double)Root->VisitCounts[I];
        }
    }
    std::cout << MaxWinRate << std::endl;
    std::cout << "    - DrawRate: " << DrawRate << std::endl;
    std::cout << "    - PV: ";
    const auto PV = currentPV(NI_Root);
    for (const auto& Move : PV) {
        std::cout << nshogi::io::sfen::move32ToSfen(Move) << " ";
    }
    std::cout << std::endl;
    std::cout << "====================" << std::endl;
}

std::vector<core::Move32> BookMaker::currentPV(NodeIndex N) const {
    std::vector<core::Move32> PV;

    std::unordered_set<NodeIndex> Visited;

    const Node* CurrentNode = &Nodes[N];

    while (true) {
        if (Visited.find(CurrentNode->Index) != Visited.end()) {
            break;
        }
        Visited.insert(CurrentNode->Index);

        // Find the most visited child.
        std::size_t BestIndex = 0;
        uint64_t MaxVisits = 0;
        float MaxPolicy = 0.0f;

        for (std::size_t I = 0; I < CurrentNode->Moves.size(); ++I) {
            if (CurrentNode->VisitCounts[I] > MaxVisits) {
                MaxVisits = CurrentNode->VisitCounts[I];
                BestIndex = I;
                MaxPolicy = CurrentNode->PolicyRaw[I];
            } else if (CurrentNode->VisitCounts[I] == MaxVisits) {
                if (CurrentNode->PolicyRaw[I] > MaxPolicy) {
                    BestIndex = I;
                    MaxPolicy = CurrentNode->PolicyRaw[I];
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
