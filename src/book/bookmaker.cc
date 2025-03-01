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
#include "../io/book.h"

#include <queue>
#include <cmath>
#include <iostream>
#include <fstream>

#include <nshogi/core/statebuilder.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace book {

BookMaker::BookMaker(const Context* Context, std::shared_ptr<logger::Logger> Logger) {
    Manager = std::make_unique<mcts::Manager>(Context, Logger);
}

void BookMaker::makeBook(uint64_t NumGenerates, const std::string& Path, const std::string& InitialPositionPath) {
    std::ofstream Ofs(Path, std::ios::out | std::ios::binary);
    if (!Ofs) {
        std::cerr << "Failed to open " << Path << std::endl;
        return;
    }

    Manager->setIsThoughtLogEnabled(true);

    core::StateConfig Config;
    Config.MaxPly = 320;
    Config.Rule = core::ER_Declare27;
    Limit L;
    L.NumNodes = 100000;

    auto SeedComparingFunction = [](const BookSeed& S1, const BookSeed& S2) {
        return S1.logProbability() < S2.logProbability();
    };
    std::priority_queue<BookSeed, std::vector<BookSeed>, decltype(SeedComparingFunction)> Queue;

    // Prepare initial position(s).
    std::ifstream InitialPositionsIfs(InitialPositionPath);
    uint64_t LoadedCount = 0;
    if (InitialPositionsIfs) {
        std::string Sfen;
        while (std::getline(InitialPositionsIfs, Sfen)) {
            if (Sfen.size() == 0 || Sfen[0] == '#') {
                continue;
            }

            core::State S = ::nshogi::io::sfen::StateBuilder::newState(Sfen);
            Queue.emplace(S, 0);
            ++LoadedCount;
        }
        std::cout << LoadedCount << " initial positions are loaded." << std::endl;
    }
    {
        core::State S = core::StateBuilder::getInitialState();
        Queue.emplace(S, 0);
    }

    std::set<core::HuffmanCode> Visited;
    uint64_t Count = 0;
    while (!Queue.empty()) {
        if (Count >= NumGenerates) {
            break;
        }

        const BookSeed Seed = Queue.top();
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

        std::cout << ::nshogi::io::sfen::stateToSfen(State) << std::endl;
        std::cout << Seed.logProbability() << std::endl;

        std::mutex Mtx;
        bool IsCallbackCalled = false;
        std::condition_variable CV;
        std::vector<std::pair<core::Move32, double>> Policy;
        double PolicyMax = 0.0;
        Manager->thinkNextMove(State, Config, L, [&](core::Move32 BestMove, std::unique_ptr<mcts::ThoughtLog> TL) {
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
                PolicyMax = std::max(PolicyMax, Policy[I].second);
            }

            BookEntry BE(Seed.huffmanCode(), BestMove, TL->WinRate, TL->DrawRate);
            io::book::writeBookEntry(Ofs, BE);
            {
                std::lock_guard<std::mutex> Lock(Mtx);
                IsCallbackCalled = true;
            }
            CV.notify_one();
        });

        {
            std::unique_lock<std::mutex> Lock(Mtx);
            CV.wait(Lock, [&]() { return IsCallbackCalled; });
        }

        // Push the next states.
        for (const auto& [Move, P] : Policy) {
            if (P < PolicyMax * 0.1) {
                continue;
            }
            const double NextLogProbability = Seed.logProbability() + std::log(P);
            State.doMove(Move);
            Queue.emplace(State, NextLogProbability, Seed.huffmanCode());
            State.undoMove();
        }

        ++Count;
        std::cout << "Progress: " << (double)Count / (double)NumGenerates * 100.0 << std::endl;
    }
}

void BookMaker::refineBook(const std::string& UnrefinedPath, const std::string& OutPath) {
    std::ifstream Ifs(UnrefinedPath, std::ios::in | std::ios::binary);
    if (!Ifs) {
        std::cerr << "Failed to open " << UnrefinedPath << std::endl;
        return;
    }

    std::ofstream Ofs(OutPath, std::ios::out | std::ios::binary);
    if (!Ofs) {
        std::cerr << "Failed to open " << OutPath << std::endl;
        return;
    }

    // Load the file.
    std::map<core::HuffmanCode, BookEntry> BookEntries;
    {
        std::vector<BookEntry> BookEntryTemp = nshogi::engine::io::book::readBook(Ifs);
        for (const auto& BE : BookEntryTemp) {
            std::cout << "\rLoading: " << BookEntries.size() << std::flush;
            BookEntries.emplace(BE.huffmanCode(), BE);
        }
        std::cout << std::endl;
    }

    uint64_t MaxIteration = 100;
    for (uint64_t Iteration = 0; Iteration < MaxIteration; ++Iteration) {
        uint64_t UpdatedCount = 0;

        for (const auto& [Huffman, Entry] : BookEntries) {
            const auto Position = core::HuffmanCode::decode(Huffman);
            auto State = core::StateBuilder::newState(Position);
            if (doMinMaxSearchOnBook(&State, BookEntries)) {
                ++UpdatedCount;
            }
        }
        std::cout << "Iteration: " << Iteration << ", UpdatedCount: " << UpdatedCount << std::endl;

        if (UpdatedCount == 0) {
            break;
        }
    }

    std::cout << "BookEntries.size(): " << BookEntries.size() << std::endl;
    for (const auto& [Huffman, Entry] : BookEntries) {
        io::book::writeBookEntry(Ofs, Entry);
    }
}

bool BookMaker::doMinMaxSearchOnBook(core::State* State, std::map<core::HuffmanCode, BookEntry>& BookEntries, double Alpha) {
    bool Updated = false;

    BookEntry& ThisEntry = BookEntries.find(core::HuffmanCode::encode(State->getPosition()))->second;

    auto NextMoves = core::MoveGenerator::generateLegalMoves(*State);
    const BookEntry* BestEntry = nullptr;
    core::Move32 BestMove = core::Move32::MoveNone();
    double BestValue = -1.0;

    for (const auto& Move : NextMoves) {
        State->doMove(Move);

        const auto ChildEntryit = BookEntries.find(core::HuffmanCode::encode(State->getPosition()));
        if (ChildEntryit != BookEntries.end()) {
            const double ChildWinRate = 1.0 - ChildEntryit->second.winRate();
            const double ChildDrawRate = ChildEntryit->second.drawRate();
            const double ChildValue = ChildWinRate * (1.0 - ChildDrawRate) + 0.5 * ChildDrawRate;
            if (ChildValue > BestValue) {
                BestValue = ChildValue;
                BestMove = Move;
                BestEntry = &ChildEntryit->second;
            }
        }

        State->undoMove();
    }

    if (BestEntry != nullptr) {
        const double ThisValue = ThisEntry.winRate() * (1.0 - ThisEntry.drawRate()) + 0.5 * ThisEntry.drawRate();
        const double NewWinRate = Alpha * ThisEntry.winRate() + (1.0 - Alpha) * (1.0 - BestEntry->winRate());
        const double NewDrawRate = Alpha * ThisEntry.drawRate() + (1.0 - Alpha) * BestEntry->drawRate();
        if (core::Move16(BestMove) == ThisEntry.bestMove()) {
            ThisEntry.setWinRate(NewWinRate);
            ThisEntry.setDrawRate(NewDrawRate);
            Updated = true;
        } else if (BestValue > ThisValue) {
            ThisEntry.setBestMove(BestMove);
            ThisEntry.setWinRate(NewWinRate);
            ThisEntry.setDrawRate(NewDrawRate);
            Updated = true;
        }
    }

    return Updated;
}

} // namespace book
} // namespace engine
} // namespace nshogi
