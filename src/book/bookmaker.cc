//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookmaker.h"
#include "../mcts/manager.h"
#include "../protocol/usilogger.h"

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/state.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/sfen.h>

#include <random>
#include <vector>

namespace nshogi {
namespace engine {
namespace book {

std::pair<core::Move32, std::unique_ptr<mcts::ThoughtLog>>
BookMaker::startThinking(core::State* State, const core::StateConfig& Config,
                         const std::vector<core::Move32>& BannedMoves,
                         const engine::Limit& Limit) {
    std::mutex Mutex;
    bool SearchFinished = false;
    std::condition_variable CV;

    core::Move32 BestMove;
    std::unique_ptr<mcts::ThoughtLog> Log;

    std::function<void(core::Move32, std::unique_ptr<mcts::ThoughtLog>)>
        Callback = [&](core::Move32 Move,
                       std::unique_ptr<mcts::ThoughtLog> ThoughtLog) {
            BestMove = Move;
            Log = std::move(ThoughtLog);

            {
                std::lock_guard<std::mutex> Lock(Mutex);
                SearchFinished = true;
            }
            CV.notify_one();
        };

    Manager->thinkNextMove(*State, Config, Limit, Callback);

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        CV.wait(Lock, [&] { return SearchFinished; });
    }

    return std::make_pair(BestMove, std::move(Log));
}

void BookMaker::evaluate(core::State* State, const core::StateConfig& Config) {
    if (State->canDeclare()) {
        return;
    }

    const auto RepeitionStatus = State->getRepetitionStatus();
    if (RepeitionStatus != core::RepetitionStatus::NoRepetition) {
        return;
    }

    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    if (Moves.size() == 0) {
        return;
    }

    engine::Limit Limit{0, 0, 30000};
    const auto& [BestMove, ThoughtLog] =
        startThinking(State, Config, {}, Limit);

    BookEntry Entry;

    if (ThoughtLog->PlyToTerminal != 0) {
        if (ThoughtLog->PlyToTerminal > 0) {
            Entry.WinRate = 1.0;
            Entry.DrawRate = 0.0;
        } else if (ThoughtLog->PlyToTerminal < 0) {
            Entry.WinRate = 0.0;
            Entry.DrawRate = 0.0;
        }
    } else {
        Entry.WinRate = ThoughtLog->WinRate;
        Entry.DrawRate = ThoughtLog->DrawRate;
    }
    Entry.BestMove = BestMove;

    const auto Sfen = nshogi::io::sfen::positionToSfen(State->getPosition());
    MyBook.update(Sfen, Entry);
}

void BookMaker::updateNegaMaxValue(core::State* State,
                                   const core::StateConfig& Config) {
    double BestScore = -1.0;
    double BestWinRate = std::numeric_limits<double>::lowest();
    double BestDrawRate = 0.0;
    core::Move32 BestMove = core::Move32::MoveNone();

    const auto MyColor = State->getSideToMove();
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    for (core::Move32 Move : Moves) {
        State->doMove(Move);

        const auto CanDeclare = State->canDeclare();
        const auto RepetitionStatus = State->getRepetitionStatus();
        const BookEntry* Entry =
            MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));
        const auto NextMoves = core::MoveGenerator::generateLegalMoves(*State);

        State->undoMove();

        double WinRate = 0.5;
        double DrawRate = 0.0;

        if (CanDeclare) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::WinRepetition ||
                   RepetitionStatus ==
                       core::RepetitionStatus::SuperiorRepetition) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::LossRepetition ||
                   RepetitionStatus ==
                       core::RepetitionStatus::InferiorRepetition) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::Repetition) {
            WinRate = (MyColor == core::Black) ? Config.BlackDrawValue
                                               : Config.WhiteDrawValue;
            DrawRate = 1.0;
        } else if (NextMoves.size() == 0) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (Entry != nullptr) {
            WinRate = 1.0 - Entry->WinRate;
            DrawRate = Entry->DrawRate;
        } else {
            continue;
        }

        const double ChildScore = (MyColor == core::Black)
                                      ? (DrawRate * Config.BlackDrawValue +
                                         (1.0 - DrawRate) * WinRate)
                                      : (DrawRate * Config.WhiteDrawValue +
                                         (1.0 - DrawRate) * WinRate);

        if (ChildScore > BestScore) {
            BestScore = ChildScore;
            BestWinRate = WinRate;
            BestDrawRate = DrawRate;
            BestMove = Move;
        }
    }

    if (!BestMove.isNone()) {
        BookEntry Entry;
        Entry.WinRate = BestWinRate;
        Entry.DrawRate = BestDrawRate;
        Entry.BestMove = BestMove;

        const auto Sfen =
            nshogi::io::sfen::positionToSfen(State->getPosition());
        MyBook.update(Sfen, Entry);
    }
}

std::optional<BookEntry>
BookMaker::updateNegaMaxValueAllInternal(core::State* State,
                                         const core::StateConfig& Config,
                                         std::set<std::string>& Fixed) {
    if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        return std::nullopt;
    }

    BookEntry ThisEntry;

    if (State->canDeclare()) {
        ThisEntry.WinRate = 1.0;
        ThisEntry.DrawRate = 0.0;
        return ThisEntry;
    }

    if (State->getPly() > Config.MaxPly) {
        ThisEntry.WinRate = 0.0;
        ThisEntry.DrawRate = 1.0;
        return ThisEntry;
    }

    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);

    if (Moves.size() == 0) {
        ThisEntry.WinRate = 0.0;
        ThisEntry.DrawRate = 0.0;
        return ThisEntry;
    }

    const auto Sfen = nshogi::io::sfen::positionToSfen(State->getPosition());

    if (MyBook.get(Sfen) == nullptr) {
        return std::nullopt;
    }
    ThisEntry = *MyBook.get(Sfen);

    if (Fixed.contains(Sfen)) {
        return ThisEntry;
    }

    std::cout << "\r" << Fixed.size() << std::flush;

    std::optional<BookEntry> BestEntry;
    double BestScore = -1.0;
    for (const core::Move32 Move : Moves) {
        State->doMove(Move);
        const auto ChildEntryOpt =
            updateNegaMaxValueAllInternal(State, Config, Fixed);
        State->undoMove();

        if (!ChildEntryOpt.has_value()) {
            continue;
        }

        const auto& ChildEntry = ChildEntryOpt.value();

        const double ChildScore =
            (State->getSideToMove() == core::Black)
                ? (ChildEntry.DrawRate * Config.BlackDrawValue +
                   (1.0 - ChildEntry.DrawRate) * (1.0 - ChildEntry.WinRate))
                : (ChildEntry.DrawRate * Config.WhiteDrawValue +
                   (1.0 - ChildEntry.DrawRate) * (1.0 - ChildEntry.WinRate));

        if (ChildScore > BestScore) {
            BookEntry NewEntry;
            NewEntry.WinRate = 1.0 - ChildEntry.WinRate;
            NewEntry.DrawRate = ChildEntry.DrawRate;
            NewEntry.BestMove = Move;
            BestEntry = NewEntry;
            BestScore = ChildScore;
        }
    }

    Fixed.insert(Sfen);

    if (BestEntry.has_value()) {
        MyBook.update(Sfen, BestEntry.value());
        return BestEntry.value();
    }

    return ThisEntry;
}

void BookMaker::updateNegaMaxValueAll(const core::StateConfig& Config) {
    std::set<std::string> Fixed;

    for (const auto& Entry : MyBook.dictionary()) {
        std::unique_ptr<core::State> State = std::make_unique<core::State>(
            nshogi::io::sfen::StateBuilder::newState(Entry.first));

        updateNegaMaxValueAllInternal(State.get(), Config, Fixed);
    }
    std::cout << std::endl;
}

void BookMaker::executeOneIteration(core::State* State,
                                    const core::StateConfig& Config) {
    if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        return;
    }

    if (State->canDeclare()) {
        return;
    }

    if (State->getPly() > Config.MaxPly) {
        return;
    }

    if (core::MoveGenerator::generateLegalMoves(*State).size() == 0) {
        return;
    }

    const BookEntry* Entry =
        MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

    if (Entry == nullptr) {
        // Evaluate this leaf.
        std::cout << "Searching at a new state." << std::endl;
        std::cout << "Sfen: " << nshogi::io::sfen::stateToSfen(*State)
                  << std::endl;
        evaluate(State, Config);
        return;
    }

    const double Score = (State->getSideToMove() == core::Black)
                             ? (Entry->DrawRate * Config.BlackDrawValue +
                                (1.0 - Entry->DrawRate) * Entry->WinRate)
                             : (Entry->DrawRate * Config.WhiteDrawValue +
                                (1.0 - Entry->DrawRate) * Entry->WinRate);

    if (Score > 0.62) {
        // If there is a promising child, follow the child.
        State->doMove(Entry->BestMove);
    } else {
        // If the child is not promising,
        // research this node without already explored moves with probability
        // epsilon and just follow the child with probability (1.0 - epsilon).
        const double Epsilon = (Entry->DrawRate > 0.99) ? 0.3 : 0.1;

        static std::random_device RD;
        static std::mt19937_64 MT(RD());
        static std::uniform_real_distribution<double> Distribution(0.0, 1.0);

        const double R = Distribution(MT);

        if (R > Epsilon) {
            State->doMove(Entry->BestMove);
        } else {
            // Search again without already expanded moves.
            std::vector<core::Move32> BannedMoves;
            const auto Moves = core::MoveGenerator::generateLegalMoves(*State);

            std::vector<core::Move32> Targets;
            std::vector<double> Weights;
            double ScoreMin = std::numeric_limits<double>::max();
            for (core::Move32 Move : Moves) {
                State->doMove(Move);

                const auto RepetitionStatus = State->getRepetitionStatus();
                const auto* ChildEntry = MyBook.get(
                    nshogi::io::sfen::positionToSfen(State->getPosition()));

                State->undoMove();

                if (Move == Entry->BestMove && ChildEntry == nullptr) {
                    Targets.push_back(Move);
                    Weights.push_back(Score);
                } else if (ChildEntry != nullptr) {
                    const double WinRate = 1.0 - ChildEntry->WinRate;
                    const double DrawRate = ChildEntry->DrawRate;
                    const double ChildScore =
                        (State->getSideToMove() == core::Black)
                            ? (DrawRate * Config.BlackDrawValue +
                               (1.0 - DrawRate) * WinRate)
                            : (DrawRate * Config.WhiteDrawValue +
                               (1.0 - DrawRate) * WinRate);

                    if (Move == Entry->BestMove || ChildScore > Score - 0.1) {
                        Targets.push_back(Move);
                        Weights.push_back(ChildScore);
                    }
                    ScoreMin = std::min(ScoreMin, ChildScore);
                }

                if (RepetitionStatus != core::RepetitionStatus::NoRepetition ||
                    ChildEntry != nullptr) {
                    BannedMoves.push_back(Move);
                }
            }

            if (Targets.size() == 0) {
                std::cerr << "Targets.size() == 0." << std::endl;
                std::cerr << "Sfen: "
                          << nshogi::io::sfen::positionToSfen(
                                 State->getPosition())
                          << std::endl;
                std::cerr << "Score: " << Score << std::endl;
                std::cerr << "Entry->BestMove: "
                          << nshogi::io::sfen::move32ToSfen(Entry->BestMove)
                          << std::endl;
                std::cerr << "Moves: ";
                for (const auto& Move : Moves) {
                    std::cerr << nshogi::io::sfen::move32ToSfen(Move) << " ("
                              << (Move == Entry->BestMove) << "), ";
                }
                std::cerr << std::endl;
                std::cerr << "BannedMoves: ";
                for (const auto& BannedMove : BannedMoves) {
                    std::cerr << nshogi::io::sfen::move32ToSfen(BannedMove)
                              << ", ";
                }
                std::cerr << std::endl;
                abort();
            }
            if (BannedMoves.size() == Moves.size()) {
                // Already tried all legal moves.
                // Hence, just follow the best move.
                State->doMove(Entry->BestMove);
            } else {
                if (ScoreMin > Score - 0.1) {
                    // MoveNone as trying a new move.
                    Targets.push_back(core::Move32::MoveNone());
                    // A new move score is assumed to be same as
                    // the minimum value of the already expanded children.
                    Weights.push_back(ScoreMin);
                }

                // Softmax.
                constexpr double SoftmaxTemperature = 0.3;
                for (auto& Weight : Weights) {
                    Weight = std::exp(Weight / SoftmaxTemperature);
                }

                std::discrete_distribution<std::size_t> D(Weights.begin(),
                                                          Weights.end());
                const std::size_t Index = D(MT);

                const auto SampledMoves = Targets[Index];

                if (SampledMoves.isNone()) {
                    // Trying a new move.
                    // Reset the search tree before starting thinking
                    // not to any banned move is selected.
                    Manager->resetSearchTree();
                    engine::Limit Limit{0, 0, 2000};
                    const auto Result =
                        startThinking(State, Config, BannedMoves, Limit);
                    assert(!Result.first.isNone());
                    State->doMove(Result.first);
                } else {
                    State->doMove(SampledMoves);
                }
            }
        }
    }

    // Go deeper.
    executeOneIteration(State, Config);

    State->undoMove();

    updateNegaMaxValue(State, Config);
}

std::vector<core::Move32> BookMaker::getPV(core::State* State,
                                           const core::StateConfig& Config) {
    if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        return {};
    }

    if (State->canDeclare()) {
        return {};
    }

    if (State->getPly() > Config.MaxPly) {
        return {};
    }

    double BestScore = 0.0;
    core::Move32 BestMove = core::Move32::MoveNone();

    const auto MyColor = State->getSideToMove();
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    for (core::Move32 Move : Moves) {
        State->doMove(Move);

        const auto CanDeclare = State->canDeclare();
        const auto RepetitionStatus = State->getRepetitionStatus();
        const BookEntry* Entry =
            MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

        State->undoMove();

        double WinRate = 0.5;
        double DrawRate = 0.0;

        if (CanDeclare) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::WinRepetition ||
                   RepetitionStatus ==
                       core::RepetitionStatus::SuperiorRepetition) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::LossRepetition ||
                   RepetitionStatus ==
                       core::RepetitionStatus::InferiorRepetition) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::Repetition) {
            WinRate = (MyColor == core::Black) ? Config.BlackDrawValue
                                               : Config.WhiteDrawValue;
            DrawRate = 1.0;
        } else if (Entry != nullptr) {
            WinRate = 1.0 - Entry->WinRate;
            DrawRate = Entry->DrawRate;
        } else {
            continue;
        }

        const double ChildScore = (MyColor == core::Black)
                                      ? (DrawRate * Config.BlackDrawValue +
                                         (1.0 - DrawRate) * WinRate)
                                      : (DrawRate * Config.WhiteDrawValue +
                                         (1.0 - DrawRate) * WinRate);

        if (ChildScore > BestScore) {
            BestScore = ChildScore;
            BestMove = Move;
        }
    }

    if (!BestMove.isNone()) {
        State->doMove(BestMove);
        std::vector<core::Move32> PV = {BestMove};
        const auto Child = getPV(State, Config);
        PV.insert(PV.end(), Child.begin(), Child.end());
        State->undoMove();
        return PV;
    } else {
        return {};
    }
}

void BookMaker::start(const std::string& Dummy) {
    CManager.setIsPonderingEnabled(false);
    CManager.setIsThoughtLogEnabled(true);
    // CManager.setMinimumThinkinTimeMilliSeconds(60 * 1000);
    CManager.setMinimumThinkinTimeMilliSeconds(1 * 1000);
    CManager.setEvalCacheMemoryMB(0);
    CManager.setAvailableMemoryMB(40 * 1024);
    CManager.setNumSearchThreads(4);
    CManager.setBatchSize(62);
    CManager.setNumEvaluationThreadsPerGPU(2);
    CManager.setNumCheckmateSearchThreads(18);

    std::shared_ptr<logger::Logger> Logger =
        std::make_shared<protocol::usi::USILogger>();

    Manager = std::make_unique<mcts::Manager>(CManager.getContext(), Logger);

    {
        std::ifstream Ifs("mybook.bin", std::ios::binary);
        if (Ifs) {
            io::book::load(MyBook, Ifs);
        }
    }

    const std::vector<std::string> InitialPositions = {
        "lr5n1/3gk2g1/2l1p2+P1/p2psp3/1p3Np1l/P1pPSP3/1P2P4/1KGS1G3/LN5R1 w "
        "BSPbn4p 1",

        // "lr5+P1/3g5/3kp4/p1lpsbB2/2n2pp1l/PppPSP3/4P4/2GSNG1p1/LNK3R2 b
        // GSPn4p 1",

        // "lr5nl/3g1kgs1/2n1pp3/p1pps4/5Np1p/P1PPSP3/1+pS1P4/1KG2G3/LN5RL b
        // BPb4p 1",

        // "lr5nl/3g1kgs1/2n1p2p1/pP2s1P1p/3P1P1P1/P1S1S3P/1G2P4/1K3G3/LN1R4L w
        // B4Pbn2p 1",

        // "lr5nl/3g1kgs1/2n1p2p1/p2Ps1P1p/1p3P1P1/P1p1S3P/1PS1P4/1KG2G3/LN5RL b
        // B3Pbnp 1",

        // "lr1k1B1n1/3g2+P2/4p4/p1lps4/5pp1l/PppPSP3/+n3P4/2KS1G1p1/LN4R2 b
        // GSNPbg4p 1",

        // "lr5+P1/3gk4/4p4/p1lp5/1Pn1Spp1b/PnGP1P1P1/1r2P4/2KS1G2+l/LN7 b
        // BG2S3Pn3p 1",

        // "lr5+P1/3gk4/4p4/p1lpsb3/1pn2pp1l/P1pPSP3/1P2P4/1KGSNG1p1/LN4R2 w
        // BGSn4p 1",
        // "lr5n1/3gk2g1/2l1p2+P1/p2psp3/1p3Np1l/P1pPSP3/1P2P4/1KGS1G3/LN5R1 w
        // BSPbn4p 1",
        // "lr5+P1/3gk4/4p4/p1lp5/1Pn1spp1b/P1GPSP1P1/4P4/1K1S1G2+l/LN7 b
        // BGS3Pr2n3p 1",
        // "lr5+P1/3gk4/4p4/p1lps4/1Pn1Npp1b/P1GPSP1P1/4P4/1K1S1G2+l/LN7 w
        // BGS3Prn3p 1",
        // "lr5+P1/3gk4/4p4/p1lpsp3/1Pn2NpB1/P1GPSP1b1/4P4/1K1S1G2+l/LN7 w
        // GSN3Pr4p 1",
        // "lr5+P1/3gk4/4p4/p1lpsp3/1Pn2Np1b/P1GPSP3/4P4/1K1S1G2+l/LN7 b
        // BGSN4Pr3p 1",
        // "lr5+P1/3gk4/4p4/p1lpsp3/1Pn2Np2/P1GPSP1b1/4P4/1K1S1G2+l/LN7 b
        // BGSN3Pr4p 1",
        // "lr5+P1/3gk4/4p4/p1lpsN3/1Pn2pp1b/P1GPSP1P1/4P4/1K1S1G2+l/LN7 w
        // BGS3Prn3p 1",
        // "lr5n1/3gk2+P1/4p4/p1lpsp1b1/1Pn2Np2/P1GPSP3/4P4/1K1S1G2+l/LN7 b
        // BGS4Pr3p 1",
        // "lr5n1/3gk2g1/2l1p2+P1/p1Ppsp1R1/1Pn2Np1b/P1GPSP3/4P4/1K1S1G2+l/LN7 w
        // BS4P2p 1",
        // "lr5n1/3gk2g1/2l1p2+P1/p2psp1R1/1Pn2Np2/P1GPSP3/4P4/1K1S1G2+l/LN7 w
        // BS5Pb2p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/2p2Np1b/PPKPSP3/1p2P4/3S1G2+l/LN7 b
        // BSNL3Pg2p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/5Np1b/PPlPSP3/1pK1P4/3S1G2+l/LN7 b
        // BSN3Pg3p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/5Np1b/PPKPSP3/1p2P4/3S1G2+l/LN7 w
        // BSNL3Pg3p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/2l2Np1b/PPPPSP3/1pK1P4/3S1G2+l/LN7 w
        // BSN3Pg2p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/2l2Np1b/PPPPSP3/1p2P4/1K1S1G2+l/LN7 b
        // BSN3Pg2p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/2l2Np1b/PPPPSP3/4P4/1K1S1G2+l/LN7 w
        // BSN3Pg3p 1",
        // "lr5n1/3gk2g1/4p2+P1/p2psp1R1/2l2Np1b/PP1PSP3/4P4/1K1S1G2+l/LN7 b
        // BSN4Pg3p 1",
        // "lr5n1/3gk2g1/2l1p2+P1/p2psp1R1/1p3Np2/P1pPSP3/1P2P4/1KGS1G2+l/LN7 w
        // BSPbn4p 1",
        // "lr5nl/3g1kgs1/2n1p2p1/p2psp1P1/1pp2Np1L/P2PSP3/1PS1P4/1KG2G3/LN5R1 w
        // BPb3p 1",
        // "lr5nl/3g1kgs1/2n1p2p1/p2psp1P1/1pp2Np1p/P2PSP3/1PS1P4/1KG2G3/LN5RL b
        // Bb3p 1",

        // "lr5n1/3gk2g1/2l1p2+P1/p2psp1R1/2n2Np2/PPpPSP3/1G2P4/1K1S1G2+l/LN7 b
        // BS3Pb3p 1",
        // "lr5nl/3g1kgs1/2n1pp1p1/p1pps3p/1p3NpPP/P1PPSP3/1PS1P4/1KG2G3/LN5RL w
        // Bbp 1",
        // "lr5n1/3gk2g1/2l1p2+P1/p2psp3/5Np1l/PppPSP3/1P2P4/1KGS1G3/LN5R1 b
        // BSPbn4p 1",
        // "lr5nl/3g1kg2/2n1ppsp1/p1pps1p1p/1p3N1P1/P1PPSPP1P/1PS1P4/1KG2G3/LN5RL
        // w Bb 1"
    };

    /* *
    {
        const std::string Sfen =
    "lr5nl/3g1kg2/2n1ppsp1/p1pps1p1p/1p3N1P1/P1PPSPP1P/1PS1P4/1KG2G3/LN5RL w Bb
    1";
        // const std::string Sfen =
    "lr1k1B1n1/3g2+P2/4p4/p1lps4/1n3pp1l/PppPSP3/G3P4/1K1S1G1p1/LN4R2 b GSNPb4p
    1"; core::StateConfig Config; Config.BlackDrawValue = 0.0;
        Config.WhiteDrawValue = 1.0;
        Config.MaxPly = 320;
        Config.Rule = core::ER_Declare27;
        std::unique_ptr<core::State> State =
            std::make_unique<core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));
        std::set<std::string> Fixed;
        updateNegaMaxValueAll(State.get(), Config, Fixed);

        const auto* Entry = MyBook.get(Sfen);

        std::cout << std::endl;
        std::cout << "Sfen: " << Sfen << std::endl;
        std::cout << "WinRate:" << Entry->WinRate << std::endl;
        std::cout << "DrawRate:" << Entry->DrawRate << std::endl;
        std::cout << "BestMove:" <<
    nshogi::io::sfen::move32ToSfen(Entry->BestMove) << std::endl;

        {
            std::ofstream Ofs("mybook.bin", std::ios::binary);
            io::book::save(MyBook, Ofs, io::book::Format::NShogi);
        }

        {
            std::ofstream Ofs("user_book1.db");
            io::book::save(MyBook, Ofs, io::book::Format::YaneuraOu);
        }
    }

    /*/
    for (uint64_t Iteration = 0;; ++Iteration) {
        if (Iteration % 50 == 0) {
            core::StateConfig Config;
            Config.BlackDrawValue = 0.0;
            Config.WhiteDrawValue = 1.0;
            Config.MaxPly = 320;
            Config.Rule = core::ER_Declare27;
            updateNegaMaxValueAll(Config);

            {
                std::ofstream Ofs("mybook.bin", std::ios::binary);
                io::book::save(MyBook, Ofs, io::book::Format::NShogi);
            }
            {
                std::ofstream Ofs("user_book1.db");
                io::book::save(MyBook, Ofs, io::book::Format::YaneuraOu);
            }
        }

        for (const auto& Sfen : InitialPositions) {
            std::unique_ptr<core::State> State = std::make_unique<core::State>(
                nshogi::io::sfen::StateBuilder::newState(Sfen));

            core::StateConfig Config;
            Config.BlackDrawValue = 0.0;
            Config.WhiteDrawValue = 1.0;
            Config.MaxPly = 320;
            Config.Rule = core::ER_Declare27;

            std::cout << "[Current root value]" << std::endl;
            const BookEntry* Entry = MyBook.get(Sfen);
            if (Entry == nullptr) {
                std::cout << "Entry does not exist!" << std::endl;
            } else {
                std::cout << "    - Book.size(): " << MyBook.size()
                          << std::endl;
                std::cout << "    - Root: " << Sfen << std::endl;
                std::cout << "    - WinRate: " << Entry->WinRate << std::endl;
                std::cout << "    - DrawRate: " << Entry->DrawRate << std::endl;
                std::cout << "    - PV: ";

                const auto PV = getPV(State.get(), Config);
                for (const core::Move32 Move : PV) {
                    std::cout << nshogi::io::sfen::move32ToSfen(Move) << " ";
                }
                std::cout << std::endl;
            }

            executeOneIteration(State.get(), Config);

            {
                std::ofstream Ofs("mybook.bin", std::ios::binary);
                io::book::save(MyBook, Ofs, io::book::Format::NShogi);
            }

            {
                std::ofstream Ofs("user_book1.db");
                io::book::save(MyBook, Ofs, io::book::Format::YaneuraOu);
            }
        }
    }
    /* */

    Manager.reset();
}

} // namespace book
} // namespace engine
} // namespace nshogi
