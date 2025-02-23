//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookmaker.h"
#include "bookseed.h"

#include <queue>
#include <set>
#include <cmath>
#include <iostream>

#include <nshogi/core/statebuilder.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace book {

BookMaker::BookMaker(const Context* Context, std::shared_ptr<logger::Logger> Logger) {
    Manager = std::make_unique<mcts::Manager>(Context, Logger);
}

void BookMaker::enumerateBookSeeds(uint64_t NumGenerates) {
    Manager->setIsThoughtLogEnabled(true);

    core::StateConfig Config;
    Config.MaxPly = 320;
    Config.Rule = core::ER_Declare27;
    Limit L;
    L.NumNodes = 1600;

    auto SeedComparingFunction = [](const BookSeed& S1, const BookSeed& S2) {
        return S1.logProbability() < S2.logProbability();
    };
    std::priority_queue<BookSeed, std::vector<BookSeed>, decltype(SeedComparingFunction)> Queue;

    // Prepare initial position(s).
    {
        core::State S = core::StateBuilder::getInitialState();
        Queue.emplace(S, 0);
    }

    std::set<core::HuffmanCode> Visited;
    uint64_t Count = 0;
    while (!Queue.empty()) {
        if (Count > NumGenerates) {
            break;
        }

        const auto Seed = Queue.top();
        Queue.pop();

        // Check duplication.
        if (Visited.contains(Seed.huffmanCode())) {
            continue;
        }
        Visited.emplace(Seed.huffmanCode());

        // Retrieve the state.
        const auto Position = core::HuffmanCode::decode(Seed.huffmanCode());
        auto State = core::StateBuilder::newState(Position);

        auto NextMoves = core::MoveGenerator::generateLegalMoves(State);
        if (NextMoves.size() == 0) {
            continue;
        }

        std::cout << io::sfen::stateToSfen(State) << std::endl;
        std::cout << Seed.logProbability() << std::endl;

        std::mutex Mtx;
        bool IsCallbackCalled = false;
        std::condition_variable CV;
        std::vector<std::pair<core::Move32, double>> Policy;
        Manager->thinkNextMove(State, Config, L, [&](core::Move32, std::unique_ptr<mcts::ThoughtLog> TL) {
            assert(TL != nullptr);
            Policy.resize(TL->VisitCounts.size());
            uint64_t VisitSum = 0;
            for (std::size_t I = 0; I < Policy.size(); ++I) {
                Policy[I].first = State.getMove32FromMove16(TL->VisitCounts[I].first);
                Policy[I].second = (double)TL->VisitCounts[I].second;
                VisitSum += TL->VisitCounts[I].second;
            }
            for (std::size_t I = 0; I < Policy.size(); ++I) {
                Policy[I].second /= (double)VisitSum;
            }

            std::lock_guard<std::mutex> Lock(Mtx);
            IsCallbackCalled = true;
            CV.notify_one();
        });

        {
            std::unique_lock<std::mutex> Lock(Mtx);
            CV.wait(Lock, [&]() { return IsCallbackCalled; });
        }

        // Push the next states.
        for (const auto& [Move, P] : Policy) {
            const double NextLogProbability = Seed.logProbability() + std::log(P);
            State.doMove(Move);
            Queue.emplace(State, NextLogProbability, Seed.huffmanCode());
            State.undoMove();
        }

        ++Count;
    }
}

} // namespace book
} // namespace engine
} // namespace nshogi
