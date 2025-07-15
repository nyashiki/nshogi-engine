//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookmaker.h"
#include "../contextmanager.h"
#include "../mcts/manager.h"
#include "../protocol/usilogger.h"

#include <nshogi/core/state.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/sfen.h>

#include <random>
#include <vector>

namespace nshogi {
namespace engine {
namespace book {

std::pair<core::Move32, std::unique_ptr<mcts::ThoughtLog>> BookMaker::startThinking(
        core::State* State,
        const core::StateConfig& Config,
        const std::vector<core::Move32>& BannedMoves
) {
    std::mutex Mutex;
    bool SearchFinished = false;
    std::condition_variable CV;

    core::Move32 BestMove;
    std::unique_ptr<mcts::ThoughtLog> Log;

    std::function<void(core::Move32, std::unique_ptr<mcts::ThoughtLog>)> Callback =
        [&](core::Move32 Move, std::unique_ptr<mcts::ThoughtLog> ThoughtLog) {

        BestMove = Move;
        Log = std::move(ThoughtLog);

        {
            std::lock_guard<std::mutex> Lock(Mutex);
            SearchFinished = true;
        }
        CV.notify_one();
    };

    // Actual thinking time is controlled by
    // MinimumThinkingTime in Context.
    engine::Limit Limit { 1, 1, 1 };

    // TODO: prohibit banned moves.
    Manager->thinkNextMove(
            *State,
            Config,
            Limit,
            Callback,
            BannedMoves);

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        CV.wait(Lock, [&]{ return SearchFinished; });
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

    const auto& [BestMove, ThoughtLog] = startThinking(State, Config, { });

    BookEntry Entry;
    Entry.WinRate = ThoughtLog->WinRate;
    Entry.DrawRate = ThoughtLog->DrawRate;
    Entry.BestMove = BestMove;

    const auto Sfen = nshogi::io::sfen::positionToSfen(State->getPosition());
    MyBook.update(Sfen, Entry);
}

void BookMaker::updateNegaMaxValue(core::State* State, const core::StateConfig& Config) {
    double BestScore = 0.0;
    double BestWinRate = 0.0;
    double BestDrawRate = 0.0;
    core::Move32 BestMove = core::Move32::MoveNone();

    const auto MyColor = State->getSideToMove();
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    for (core::Move32 Move : Moves) {
        State->doMove(Move);

        const auto CanDeclare = State->canDeclare();
        const auto RepetitionStatus = State->getRepetitionStatus();
        const BookEntry* Entry = MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

        State->undoMove();

        double WinRate = 0.5;
        double DrawRate = 0.0;

        if (CanDeclare) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::WinRepetition ||
                RepetitionStatus == core::RepetitionStatus::SuperiorRepetition) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::LossRepetition ||
                RepetitionStatus == core::RepetitionStatus::InferiorRepetition) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::Repetition) {
            WinRate = (MyColor == core::Black) ? Config.BlackDrawValue : Config.WhiteDrawValue;
            DrawRate = 1.0;
        } else if (Entry != nullptr) {
            WinRate = 1.0 - Entry->WinRate;
            DrawRate = Entry->DrawRate;
        } else {
            continue;
        }

        const double ChildScore = (MyColor == core::Black)
            ? (DrawRate * Config.BlackDrawValue + (1.0 - DrawRate) * WinRate)
            : (DrawRate * Config.WhiteDrawValue + (1.0 - DrawRate) * WinRate);

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

        const auto Sfen = nshogi::io::sfen::positionToSfen(State->getPosition());
        MyBook.update(Sfen, Entry);
    }
}

void BookMaker::executeOneIteration(core::State* State, const core::StateConfig& Config) {
    if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        return;
    }

    if (State->canDeclare()) {
        return;
    }

    if (State->getPly() > Config.MaxPly) {
        return;
    }

    const BookEntry* Entry = MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

    if (Entry == nullptr) {
        // Evaluate this leaf.
        std::cout << "Searching at a new state." << std::endl;
        std::cout << "Sfen: " <<  nshogi::io::sfen::stateToSfen(*State) << std::endl;
        evaluate(State, Config);
        return;
    }

    const double Score = (State->getSideToMove() == core::Black)
        ? (Entry->DrawRate * Config.BlackDrawValue + (1.0 - Entry->DrawRate) * Entry->WinRate)
        : (Entry->DrawRate * Config.WhiteDrawValue + (1.0 - Entry->DrawRate) * Entry->WinRate);

    if (Score > 0.5) {
        // If there is a promising child, follow the child.
        State->doMove(Entry->BestMove);
    } else {
        // If the child is not promising,
        // research this node without already explored moves with probability epsilon
        // and just follow the child with probability (1.0 - epsilon).
        // TODO: if score is near 0.5 then we should prioritize
        // going deeper than searching broader?
        // However, sometimes actually better PV is predicted much worse
        // especially with small search budget.
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
                const auto* ChildEntry =
                    MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

                State->undoMove();

                if (ChildEntry != nullptr) {
                    const double WinRate = 1.0 - ChildEntry->WinRate;
                    const double DrawRate = ChildEntry->DrawRate;
                    const double ChildScore = (State->getSideToMove() == core::Black)
                        ? (DrawRate * Config.BlackDrawValue + (1.0 - DrawRate) * WinRate)
                        : (DrawRate * Config.WhiteDrawValue + (1.0 - DrawRate) * WinRate);
                    Targets.push_back(Move);
                    Weights.push_back(ChildScore);
                    ScoreMin = std::min(ScoreMin, ChildScore);
                }

                if (RepetitionStatus != core::RepetitionStatus::NoRepetition || ChildEntry != nullptr) {
                    BannedMoves.push_back(Move);
                }
            }

            if (BannedMoves.size() == Moves.size()) {
                // Already tried all legal moves.
                // Hence, just follow the best move.
                State->doMove(Entry->BestMove);
            } else {
                // MoveNone as trying a new move.
                Targets.push_back(core::Move32::MoveNone());
                // A new move score is assumed to be same as
                // the minimum value of the already expanded children.
                Weights.push_back(ScoreMin);

                // Softmax.
                constexpr double SoftmaxTemperature = 0.1;
                for (auto& Weight : Weights) {
                    Weight = std::exp(Weight / SoftmaxTemperature);
                }

                std::discrete_distribution<std::size_t> D(Weights.begin(), Weights.end());
                const std::size_t Index = D(MT);

                const auto SampledMoves = Targets[Index];

                if (SampledMoves.isNone()) {
                    // Trying a new move.
                    const auto Result = startThinking(State, Config, BannedMoves);
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

std::vector<core::Move32> BookMaker::getPV(core::State* State, const core::StateConfig& Config) {
    if (State->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        return { };
    }

    if (State->canDeclare()) {
        return { };
    }

    if (State->getPly() > Config.MaxPly) {
        return { };
    }

    double BestScore = 0.0;
    core::Move32 BestMove = core::Move32::MoveNone();

    const auto MyColor = State->getSideToMove();
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    for (core::Move32 Move : Moves) {
        State->doMove(Move);

        const auto CanDeclare = State->canDeclare();
        const auto RepetitionStatus = State->getRepetitionStatus();
        const BookEntry* Entry = MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

        State->undoMove();

        double WinRate = 0.5;
        double DrawRate = 0.0;

        if (CanDeclare) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::WinRepetition ||
                RepetitionStatus == core::RepetitionStatus::SuperiorRepetition) {
            WinRate = 0.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::LossRepetition ||
                RepetitionStatus == core::RepetitionStatus::InferiorRepetition) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (RepetitionStatus == core::RepetitionStatus::Repetition) {
            WinRate = (MyColor == core::Black) ? Config.BlackDrawValue : Config.WhiteDrawValue;
            DrawRate = 1.0;
        } else if (Entry != nullptr) {
            WinRate = 1.0 - Entry->WinRate;
            DrawRate = Entry->DrawRate;
        } else {
            continue;
        }

        const double ChildScore = (MyColor == core::Black)
            ? (DrawRate * Config.BlackDrawValue + (1.0 - DrawRate) * WinRate)
            : (DrawRate * Config.WhiteDrawValue + (1.0 - DrawRate) * WinRate);

        if (ChildScore > BestScore) {
            BestScore = ChildScore;
            BestMove = Move;
        }
    }

    if (!BestMove.isNone()) {
        State->doMove(BestMove);
        std::vector<core::Move32> PV = { BestMove };
        const auto Child = getPV(State, Config);
        PV.insert(PV.end(), Child.begin(), Child.end());
        State->undoMove();
        return PV;
    } else {
        return { };
    }
}

void BookMaker::start(const std::string& Sfen) {
    ContextManager CManager;

    CManager.setIsPonderingEnabled(false);
    CManager.setIsThoughtLogEnabled(true);
    // CManager.setMinimumThinkinTimeMilliSeconds(60 * 1000);
    CManager.setMinimumThinkinTimeMilliSeconds(3 * 1000);
    CManager.setEvalCacheMemoryMB(0);
    CManager.setAvailableMemoryMB(40 * 1024);
    CManager.setNumSearchThreads(3);
    CManager.setNumEvaluationThreadsPerGPU(2);
    CManager.setNumCheckmateSearchThreads(4);

    std::shared_ptr<logger::Logger> Logger = std::make_shared<protocol::usi::USILogger>();

    Manager = std::make_unique<mcts::Manager>(CManager.getContext(), Logger);

    {
        std::ifstream Ifs("mybook.bin", std::ios::binary);
        if (Ifs) {
            io::book::load(MyBook, Ifs);
        }
    }

    while (true) {
        std::unique_ptr<core::State> State =
            std::make_unique<core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));

        core::StateConfig Config;
        Config.BlackDrawValue = 0.0;
        Config.WhiteDrawValue = 1.0;

        std::cout << "[Current root value]" << std::endl;
        const BookEntry* Entry = MyBook.get(Sfen);
        if (Entry == nullptr) {
            std::cout << "Entry does not exist!" << std::endl;
        } else {
            std::cout << "    - Book.size(): " << MyBook.size() << std::endl;
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

        std::ofstream Ofs("mybook.bin", std::ios::binary);
        io::book::save(MyBook, Ofs);
    }

    Manager.reset();
}

} // namespace book
} // namespace engine
} // namespace nshogi
