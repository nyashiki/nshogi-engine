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
#include "../io/book.h"

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

static std::unordered_map<std::string, std::unordered_set<uint32_t>> Precedents;
static BookMaker Restore;

constexpr uint64_t SimulationMax = 500000;
constexpr uint64_t SimulationMin = 50000;

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

    {
        std::ifstream IndexIfs("book_ckpt_bak/ckpt_latest.index", std::ios::binary);
        std::ifstream DataIfs("book_ckpt_bak/ckpt_latest.data", std::ios::binary);
        if (IndexIfs && DataIfs) {
            io::book::load(&Restore, IndexIfs, DataIfs);
            std::cout << "[BookMaker] Loaded restore checkpoint from book_ckpt_45kei_plain/ckpt_latest." << std::endl;
        }
    }
}

void BookMaker::start(const core::State& RootState, uint64_t NumSimulations) {
    // TODO: resume with a non-root state.

    {
        std::ifstream PrecedentsIfs("precedent.sfen");

        std::string Line;

        uint64_t Loaded = 0;
        while (std::getline(PrecedentsIfs, Line)) {
            std::cout << "\r[BookMaker] Loading precedent line " << Loaded++ << std::flush;

            if (Line == "" || Line[0] == '#') {
                continue;
            }

            const auto State = nshogi::io::sfen::StateBuilder::newState(Line);
            auto Replay = nshogi::core::StateBuilder::newState(State.getInitialPosition());

            while (Replay.getPly() < State.getPly()) {
                const auto Move = State.getHistoryMove(Replay.getPly());
                const auto Sfen = nshogi::io::sfen::positionToSfen(Replay.getPosition());
                Precedents[Sfen].insert(Move.value());
                Replay.doMove(Move);
            }
        }
        std::cout << std::endl;
    }

    prepareMCTSManager();
    prepareRoot(RootState);

    for (uint64_t Simulation = 0; Simulation < NumSimulations; ++Simulation) {
        auto [State, NodeToSearch] = computeNodeToSearch();

        // if (State == nullptr) {
        if (true) {
            State = std::make_unique<core::State>(RootState.clone());
            NodeToSearch = findNode(*State);
        }

        // Step1: Select one leaf node.
        auto [Trajectory, LeafWinRate, LeafDrawRate] = collectOneLeaf(State.get());
        const auto LeafNodeIndex = Trajectory.back().first;

        // Step2: Expand the leaf node and evaluate it.
        if (Nodes[LeafNodeIndex].visitCount() == 1) {
            std::cout << "[BookMaker] Evaluating state: "
                      << nshogi::io::sfen::stateToSfen(*State) << std::endl;

            expandAndEvaluate(State.get(), LeafNodeIndex);
            LeafWinRate = Nodes[LeafNodeIndex].WinRateRaw;
            LeafDrawRate = Nodes[LeafNodeIndex].DrawRateRaw;
        }

        // Step3: Backpropagate the evaluation result.
        Trajectory.pop_back();
        backpropagate(Trajectory, LeafWinRate, LeafDrawRate);

        if (Simulation % 1 == 0) {
            const NodeIndex Root = findNode(RootState);
            outputDebugInfo(Root);
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

            {
                const std::string BookPath = "user_book1.db";
                std::ofstream BookOfs(BookPath);
                io::book::save(book(), io::book::Format::YaneuraOu, BookOfs);
            }

            std::cout << "[BookMaker] Checkpointed at simulation " << Simulation << "." << std::endl;
        }
    }
}

std::vector<BookEntry> BookMaker::book() const {
    std::vector<BookEntry> Entries;

    for (const auto& [Sfen, NodeIdx] : NodeIndices) {
        const Node& N = Nodes[NodeIdx];

        // Find the most visited move.
        std::size_t BestIndex = 0;
        uint64_t MaxVisits = 0;

        for (std::size_t I = 0; I < N.Moves.size(); ++I) {
            if (N.VisitCounts[I] > MaxVisits) {
                MaxVisits = N.VisitCounts[I];
                BestIndex = I;
            }
        }

        if (MaxVisits == 0) {
            continue;
        }

        BookEntry Entry;
        Entry.Sfen = Sfen;
        Entry.Move = N.Moves[BestIndex];
        Entry.WinRate = (float)(N.WinRateAccumulateds[BestIndex] / (double)N.VisitCounts[BestIndex]);
        Entry.DrawRate = (float)(N.DrawRateAccumulateds[BestIndex] / (double)N.VisitCounts[BestIndex]);

        Entries.push_back(Entry);
    }

    return Entries;
}

void BookMaker::prepareMCTSManager() {
    MCTSManager.reset();

    CManager.setIsPonderingEnabled(false);
    CManager.setMinimumThinkinTimeMilliSeconds(2 * 1000);
    CManager.setMaximumThinkinTimeMilliSeconds(2 * 1000);
    CManager.setNumCheckmateSearchThreads(8);
    CManager.setBookEnabled(false);
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
    if (Nodes.empty()) {
        // Nodes[0] is the null node.
        Nodes.emplace_back(NI_Null);
    }

    const auto Key = nshogi::io::sfen::positionToSfen(RootState.getPosition());
    if (NodeIndices.find(Key) == NodeIndices.end()) {
        NodeIndex Root = (NodeIndex)Nodes.size();
        Nodes.emplace_back(Root);
        NodeIndices[nshogi::io::sfen::positionToSfen(RootState.getPosition())] = Root;
    }
}

NodeIndex BookMaker::findNode(const core::State& State) const {
    const auto It = NodeIndices.find(nshogi::io::sfen::positionToSfen(State.getPosition()));
    if (It != NodeIndices.end()) {
        return It->second;
    }

    throw std::runtime_error("[BookMaker] Error: Node not found for the given state.");
}

std::pair<std::unique_ptr<core::State>, NodeIndex> BookMaker::computeNodeToSearch() {
    // The most largest difference between most visited Q(s, a) and max Q(s, a) in
    // NodeIndices.
    double MaxDiff = std::numeric_limits<double>::lowest();
    NodeIndex NodeToSearch = NI_Null;
    std::unique_ptr<core::State> StateToSearch = nullptr;

    for (const auto& [Sfen, NodeIdx] : NodeIndices) {
        const Node& N = Nodes[NodeIdx];

        if (N.visitCount() <= 1) {
            continue;
        }

        uint64_t MostVisitedCount = 0;
        double MaxQ = std::numeric_limits<double>::lowest();
        double MostVisitedQ = 0.0;
        for (std::size_t I = 0; I < N.Moves.size(); ++I) {
            if (N.VisitCounts[I] == 0) {
                continue;
            }

            if (N.VisitCounts[I] > MostVisitedCount) {
                MostVisitedCount = N.VisitCounts[I];
                MostVisitedQ = N.WinRateAccumulateds[I] / (double)N.VisitCounts[I];
            }

            const double Q = N.WinRateAccumulateds[I] / (double)N.VisitCounts[I];
            if (Q > MaxQ) {
                MaxQ = Q;
            }
        }

        const double Diff = MaxQ - MostVisitedQ;
        if (Diff > MaxDiff) {
            MaxDiff = Diff;
            NodeToSearch = NodeIdx;
            StateToSearch = std::make_unique<core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));
        }
    }

    return { std::move(StateToSearch), NodeToSearch };
}

std::tuple<std::vector<std::pair<NodeIndex, std::size_t>>, float, float> BookMaker::collectOneLeaf(core::State* State) {
    std::vector<std::pair<NodeIndex, std::size_t>> Trajectory;

    NodeIndex CurrentNode = findNode(*State);

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
            return { std::move(Trajectory), 0.5f, 1.0f };
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
        Nodes[N].WinRateRaw = 0.5f;
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

void BookMaker::evaluate(core::State* State, NodeIndex N) {
    // Temporary.
    // If Restore has valid entry, retore the evaluation result from it.
    {
        const auto It = Restore.NodeIndices.find(nshogi::io::sfen::positionToSfen(State->getPosition()));
        if (It != Restore.NodeIndices.end()) {
            const NodeIndex RestoreNodeIndex = It->second;

            bool IsValid = true;

#ifndef NDEBUG
            // nan, inf check.
            if (!std::isfinite(Restore.Nodes[RestoreNodeIndex].WinRateRaw) ||
                !std::isfinite(Restore.Nodes[RestoreNodeIndex].DrawRateRaw)) {
                IsValid = false;
            }
            // for policy.
            for (const auto& P : Restore.Nodes[RestoreNodeIndex].PolicyRaw) {
                if (!std::isfinite(P)) {
                    IsValid = false;
                    break;
                }
            }
#endif

            if (IsValid) {
                Nodes[N].PolicyRaw.resize(Nodes[N].Moves.size(), 0.0f);
                Nodes[N].VisitCounts.resize(Nodes[N].Moves.size(), 0);
                Nodes[N].WinRateAccumulateds.resize(Nodes[N].Moves.size(), 0.0);
                Nodes[N].DrawRateAccumulateds.resize(Nodes[N].Moves.size(), 0.0);
                Nodes[N].Children.resize(Nodes[N].Moves.size(), NI_Null);

                Nodes[N].WinRateRaw = Restore.Nodes[RestoreNodeIndex].WinRateRaw;
                Nodes[N].DrawRateRaw = Restore.Nodes[RestoreNodeIndex].DrawRateRaw;

                for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
                    for (std::size_t J = 0; J < Restore.Nodes[RestoreNodeIndex].Moves.size(); ++J) {
                        if (Nodes[N].Moves[I] == Restore.Nodes[RestoreNodeIndex].Moves[J]) {
                            Nodes[N].PolicyRaw[I] = Restore.Nodes[RestoreNodeIndex].PolicyRaw[J];
                            break;
                        }
                    }
                }

                std::cout << "[BookMaker] Restored evaluation." << std::endl;
                return;
            }
        }
    }

    std::mutex Mutex;
    std::condition_variable CV;
    bool ThinkingDone = false;

    auto STCallback = [&](mcts::Tree* Tree) {
        storeSearchResult(State, Tree->getRoot(), true, N);

        std::lock_guard<std::mutex> Lock(Mutex);
        ThinkingDone = true;
        CV.notify_one();
    };

    engine::Limit Limit { 0, 0, 0, SimulationMax };
    MCTSManager->thinkNextMove(
        *State,
        StateConfig,
        Limit,
        nullptr,
        STCallback
    );

    { // Wait until evaluation is done.
        std::unique_lock<std::mutex> Lock(Mutex);
        CV.wait(Lock, [&ThinkingDone]() { return ThinkingDone; });
    }
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

        assert(std::isfinite(Nodes[NodeIdx].WinRateAccumulateds[MoveIdx]));
        assert(std::isfinite(Nodes[NodeIdx].DrawRateAccumulateds[MoveIdx]));
    }
}

void BookMaker::storeSearchResult(core::State* State, mcts::Node* N, bool IsRoot, NodeIndex RootNodeIndex) {
    if (!IsRoot) {
        if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
            return;
        }

        if (N->getVisitsAndVirtualLoss() < SimulationMin) {
            return;
        }

        if (N->getNumChildren() == 0) {
            return;
        }
    }

    Node* Target = nullptr;

    if (IsRoot) {
        Target = &Nodes[RootNodeIndex];
    } else {
        const auto Key = nshogi::io::sfen::positionToSfen(State->getPosition());
        if (Restore.NodeIndices.find(Key) == Restore.NodeIndices.end()) {
            NodeIndex NewNodeIndex = (NodeIndex)(Restore.Nodes.size());
            Restore.Nodes.emplace_back(NewNodeIndex);
            Restore.NodeIndices[Key] = NewNodeIndex;
            Target = &Restore.Nodes[NewNodeIndex];
        }
    }

    if (Target != nullptr) {
        Target->Moves.resize(N->getNumChildren());
        Target->PolicyRaw.resize(N->getNumChildren(), 0.0f);
        Target->VisitCounts.resize(N->getNumChildren(), 0);
        Target->WinRateAccumulateds.resize(N->getNumChildren(), 0.0);
        Target->DrawRateAccumulateds.resize(N->getNumChildren(), 0.0);
        Target->Children.resize(N->getNumChildren(), NI_Null);

        Target->WinRateRaw = (float)(N->getWinRateAccumulated() / (double)N->getVisitsAndVirtualLoss());
        Target->DrawRateRaw = (float)(N->getDrawRateAccumulated() / (double)N->getVisitsAndVirtualLoss());
        for (std::size_t I = 0; I < N->getNumChildren(); ++I) {
            Target->Moves[I] = State->getMove32FromMove16(N->getEdge()[I].getMove());
            mcts::Node* Child = N->getEdge()[I].getTarget();
            if (Child != nullptr) {
                Target->PolicyRaw[I] = (float)Child->getVisitsAndVirtualLoss() / (float)(N->getVisitsAndVirtualLoss() - 1);
            } else {
                Target->PolicyRaw[I] = 0.0f;
            }
        }
    }

    for (std::size_t I = 0; I < N->getNumChildren(); ++I) {
        auto& Edge = N->getEdge()[I];
        mcts::Node* Child = Edge.getTarget();

        if (Child == nullptr) {
            continue;
        }

        const auto Move = State->getMove32FromMove16(Edge.getMove());
        State->doMove(Move);
        storeSearchResult(State, Child, false, RootNodeIndex);
        State->undoMove();
    }
}

std::pair<std::size_t, core::Move32> BookMaker::computeUCBMaxChild(core::State* State, NodeIndex N) {
    // TODO: draw value.

    const uint64_t ThisVisitCount = Nodes[N].visitCount();

    const auto PrecedentIt = Precedents.find(nshogi::io::sfen::positionToSfen(State->getPosition()));

    std::size_t BestIndex = 0;
    core::Move32 BestMove = core::Move32::MoveNone();
    double UCBMaxValue = -1.0;
    // std::size_t PrecedentBestIndex = 0;
    // core::Move32 PrecedentBestMove = core::Move32::MoveNone();
    // double PrecedentUCBMaxValue = -1.0;

    static constexpr int32_t CBase = 19652;
    static constexpr double CInit = 1.25;
    const double Const =
        (std::log((double)(ThisVisitCount + CBase) / (double)CBase) + CInit) *
        std::sqrt((double)ThisVisitCount);

    // const bool PrecedentOnly = State->getSideToMove() == core::Color::Black;

    for (std::size_t I = 0; I < Nodes[N].Moves.size(); ++I) {
        if (Nodes[N].visitCount() > 1) {
            if (PrecedentIt != Precedents.end()) {
                if (Nodes[N].VisitCounts[I] == 0) {
                    if (PrecedentIt->second.contains(Nodes[N].Moves[I].value())) {
                        BestIndex = I;
                        BestMove = Nodes[N].Moves[I];
                        break;
                    }
                }
            }
        }

        const double WinRate = (Nodes[N].VisitCounts[I] == 0)
                                 ? 0.0
                                 : (Nodes[N].WinRateAccumulateds[I] / (double)Nodes[N].VisitCounts[I]);
        const double DrawRate = (Nodes[N].VisitCounts[I] == 0)
                                  ? 0.0
                                  : (Nodes[N].DrawRateAccumulateds[I] / (double)Nodes[N].VisitCounts[I]);
        const double DrawValue = (State->getSideToMove() == core::Color::Black)
                                     ? 0.0 : 1.0;

        const double Q = (1 - DrawRate) * WinRate + DrawRate * DrawValue;
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

        if (State->getRepetitionStatus() == core::RepetitionStatus::NoRepetition) {
            // If the position has not been registered yet, register it.
            const auto Key = nshogi::io::sfen::positionToSfen(State->getPosition());
            if (auto It = NodeIndices.find(Key); It == NodeIndices.end()) {
                NodeIndices[Key] = Nodes[N].Children[BestIndex];
            }
        }

        State->undoMove();
    }

    assert(!BestMove.isNone());
    return { BestIndex, BestMove };
}

void BookMaker::outputDebugInfo(NodeIndex Root) const {
    std::cout << "====================" << std::endl;

    std::cout << "[Stats]" << std::endl;
    std::cout << "    - total nodes: " << Nodes.size() << std::endl;
    std::cout << "    - index size: " << NodeIndices.size() << std::endl;

    std::cout << "[Root]" << std::endl;
    std::cout << "    - visits: " << Nodes[Root].visitCount() << std::endl;

    uint64_t MaxVisitCount = 0;
    double WinRate = 0.0;
    double DrawRate = 0.0;
    for (std::size_t I = 0; I < Nodes[Root].Moves.size(); ++I) {
        if (Nodes[Root].VisitCounts[I] == 0) {
            continue;
        }

        if (Nodes[Root].VisitCounts[I] > MaxVisitCount) {
            MaxVisitCount = Nodes[Root].VisitCounts[I];
            WinRate = Nodes[Root].WinRateAccumulateds[I] / (double)Nodes[Root].VisitCounts[I];
            DrawRate = Nodes[Root].DrawRateAccumulateds[I] / (double)Nodes[Root].VisitCounts[I];
        }
    }
    std::cout << "    - winRate: " << WinRate << std::endl;
    std::cout << "    - drawRate: " << DrawRate << std::endl;
    std::cout << "    - pv: ";
    const auto& [PV, Trajectory] = currentPV(Root);
    for (const auto& Move : PV) {
        std::cout << nshogi::io::sfen::move32ToSfen(Move) << " ";
    }
    std::cout << std::endl;

    std::cout << "    - [pv]" << std::endl;
    bool Flip = false;
    for (std::size_t I = 0; I < Trajectory.size(); ++I) {
        const NodeIndex NodeIdx = Trajectory[I];
        const auto& Node = Nodes[NodeIdx];

        std::size_t SelectedMoveIndex = 0;
        for (std::size_t J = 0; J < Node.Moves.size(); ++J) {
            if (Node.Moves[J] == PV[I]) {
                SelectedMoveIndex = J;
                break;
            }
        }

        double MoveWinRate = (Node.WinRateAccumulateds[SelectedMoveIndex] / (double)Node.VisitCounts[SelectedMoveIndex]);
        if (Flip) {
            MoveWinRate = 1.0 - MoveWinRate;
        }
        std::cout << "        " << nshogi::io::sfen::move32ToSfen(PV[I])
            << ": visits=" << Node.VisitCounts[SelectedMoveIndex]
            << ", win rate=" << MoveWinRate
            << ", win rate (raw)=" << (Flip ? (1.0 - Node.WinRateRaw) : Node.WinRateRaw)
            << ", draw rate=" << (Node.DrawRateAccumulateds[SelectedMoveIndex] / (double)Node.VisitCounts[SelectedMoveIndex])
            << ", draw rate (raw)=" << Node.DrawRateRaw
            << ", policy=" << Node.PolicyRaw[SelectedMoveIndex]
            << ", improved policy=" << (float)Node.VisitCounts[SelectedMoveIndex] / (float)(Node.visitCount() - 1)
            << std::endl;

        Flip = !Flip;
    }

    std::cout << "====================" << std::endl;
}

std::pair<std::vector<core::Move32>, std::vector<NodeIndex>> BookMaker::currentPV(NodeIndex N) const {
    std::vector<core::Move32> PV;
    std::vector<NodeIndex> Trajectory;

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
        Trajectory.push_back(CurrentNode->Index);
        CurrentNode = &Nodes[CurrentNode->Children[BestIndex]];
    }

    return { std::move(PV), std::move(Trajectory) };
}

} // namespace book
} // namespace engine
} // namespace nshogi
