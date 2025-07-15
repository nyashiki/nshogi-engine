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

namespace nshogi {
namespace engine {
namespace book {

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

    std::mutex Mutex;
    bool SearchFinished = false;
    std::condition_variable CV;

    std::function<void(core::Move32, std::unique_ptr<mcts::ThoughtLog>)> Callback =
        [&](core::Move32 BestMove, std::unique_ptr<mcts::ThoughtLog> ThoughtLog) {

        BookEntry Entry;
        Entry.WinRate = ThoughtLog->WinRate;
        Entry.DrawRate = ThoughtLog->DrawRate;
        Entry.BestMove = BestMove;

        const auto Sfen = io::sfen::positionToSfen(State->getPosition());
        MyBook.update(Sfen, Entry);

        {
            std::lock_guard<std::mutex> Lock(Mutex);
            SearchFinished = true;
        }
        CV.notify_one();
    };

    // Actual thinking time is controlled by
    // MinimumThinkingTime in Context.
    engine::Limit Limit { 1, 1, 1 };

    Manager->thinkNextMove(
            *State,
            Config,
            Limit,
            Callback);

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        CV.wait(Lock, [&]{ return SearchFinished; });
    }
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
        const BookEntry* Entry = MyBook.get(io::sfen::positionToSfen(State->getPosition()));

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
        } else if (Entry != nullptr) {
            WinRate = 1.0 - Entry->WinRate;
            DrawRate = Entry->DrawRate;
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
    const BookEntry* Entry = MyBook.get(nshogi::io::sfen::positionToSfen(State->getPosition()));

    if (Entry == nullptr) {
        // Evaluate this leaf.
        evaluate(State, Config);
        return;
    }

    const double Score = (State->getSideToMove() == core::Black)
        ? (Entry->DrawRate * Config.BlackDrawValue + (1.0 - Entry->DrawRate) * (1.0 - Entry->WinRate))
        : (Entry->DrawRate * Config.WhiteDrawValue + (1.0 - Entry->DrawRate) * (1.0 - Entry->WinRate));

    if (Score > 0.5) {
        // If there is a promising child, follow the child.
        State->doMove(Entry->BestMove);
    } else {
        // If the child is not promising,
        // research this node without already explored moves with probability epsilon
        // and just follow the child with probability (1.0 - epsilon).
        const double Epsilon = 0.1;
        const double R = 0.5;

        if (R > Epsilon) {
            State->doMove(Entry->BestMove);
        } else {
            // TODO: research banning already expanded moves.
        }
    }

    // Go deeper.
    executeOneIteration(State, Config);

    State->undoMove();

    updateNegaMaxValue(State, Config);
}

void BookMaker::start(const std::string& Sfen) {
    ContextManager CManager;

    CManager.setIsThoughtLogEnabled(true);
    CManager.setMinimumThinkinTimeMilliSeconds(60 * 1000);
    CManager.setEvalCacheMemoryMB(0);

    std::shared_ptr<logger::Logger> Logger = std::make_shared<protocol::usi::USILogger>();

    Manager = std::make_unique<mcts::Manager>(CManager.getContext(), Logger);

    while (true) {
        std::unique_ptr<core::State> State =
            std::make_unique<core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));

        core::StateConfig Config;
        Config.BlackDrawValue = 0.5;
        Config.WhiteDrawValue = 0.5;

        executeOneIteration(State.get(), Config);
    }

    Manager.reset();
}

} // namespace book
} // namespace engine
} // namespace nshogi
