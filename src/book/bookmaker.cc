//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookentry.h"
#include "bookmaker.h"
#include "bookseed.h"

#include <queue>
#include <set>
#include <cmath>
#include <iostream>
#include <fstream>

#include <nshogi/core/statebuilder.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace book {

namespace {

void writeSeed(std::ofstream& Ofs, const BookSeed& BS) {
    double LP = BS.logProbability();
    bool HP = BS.hasParent();

    Ofs.write(BS.huffmanCode().data(),  (long)core::HuffmanCode::size());
    Ofs.write(BS.parentCode().data(),  (long)core::HuffmanCode::size());
    Ofs.write(reinterpret_cast<const char*>(&LP), sizeof(LP));
    Ofs.write(reinterpret_cast<const char*>(&HP), sizeof(HP));
}

BookSeed readSeed(std::ifstream& Ifs) {
    char HC[32];
    char PC[32];
    double LP;
    bool HP;

    Ifs.read(HC, (long)core::HuffmanCode::size());
    Ifs.read(PC, (long)core::HuffmanCode::size());
    Ifs.read(reinterpret_cast<char*>(&LP), sizeof(LP));
    Ifs.read(reinterpret_cast<char*>(&HP), sizeof(HP));

    return BookSeed(HC, PC, LP, HP);
}

void writeBookEntry(std::ofstream& Ofs, const BookEntry& BE) {
    core::Move16 BestMove = BE.bestMove();
    double WinRate = BE.winRate();
    double DrawRate = BE.drawRate();

    Ofs.write(BE.huffmanCode().data(),  (long)core::HuffmanCode::size());
    Ofs.write(reinterpret_cast<const char*>(&BestMove), sizeof(BestMove));
    Ofs.write(reinterpret_cast<const char*>(&WinRate), sizeof(WinRate));
    Ofs.write(reinterpret_cast<const char*>(&DrawRate), sizeof(DrawRate));
}

} // namespace

BookMaker::BookMaker(const Context* Context, std::shared_ptr<logger::Logger> Logger) {
    Manager = std::make_unique<mcts::Manager>(Context, Logger);
}

void BookMaker::enumerateBookSeeds(uint64_t NumGenerates, const std::string& Path) {
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
        writeSeed(Ofs, Seed);

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

void BookMaker::makeBookFromBookSeed(const std::string& BookSeedPath, const std::string& OutPath) {
    std::ifstream Ifs(BookSeedPath, std::ios::in | std::ios::binary);
    if (!Ifs) {
        std::cerr << "Failed to open " << BookSeedPath << std::endl;
        return;
    }

    std::ofstream Ofs(OutPath, std::ios::out | std::ios::binary);
    if (!Ofs) {
        std::cerr << "Failed to open " << OutPath << std::endl;
        return;
    }

    Ifs.seekg(0, std::ios::end);
    std::streampos FileSize = Ifs.tellg();
    const uint64_t SeedCount = (uint64_t)FileSize / sizeof(BookSeed);

    Manager->setIsThoughtLogEnabled(true);

    core::StateConfig Config;
    Config.MaxPly = 320;
    Config.Rule = core::ER_Declare27;
    Limit L;
    L.NumNodes = 10000;

    // Think all positions registered in the seed.
    Ifs.seekg(0, std::ios::beg);
    for (std::size_t I = 0; I < SeedCount; ++I) {
        BookSeed Seed = readSeed(Ifs);

        const auto Position = core::HuffmanCode::decode(Seed.huffmanCode());
        auto State = core::StateBuilder::newState(Position);

        std::mutex Mtx;
        bool IsCallbackCalled = false;
        std::condition_variable CV;
        Manager->thinkNextMove(State, Config, L, [&](core::Move32 BestMove, std::unique_ptr<mcts::ThoughtLog> TL) {
            assert(TL != nullptr);

            BookEntry BE(Seed.huffmanCode(), BestMove, TL->WinRate, TL->DrawRate);
            writeBookEntry(Ofs, BE);

            std::lock_guard<std::mutex> Lock(Mtx);
            IsCallbackCalled = true;
            CV.notify_one();
        });

        {
            std::unique_lock<std::mutex> Lock(Mtx);
            CV.wait(Lock, [&]() { return IsCallbackCalled; });
        }

        std::cout << "Progress: " << (double)(I + 1) / (double)SeedCount * 100.0 << std::endl;
    }
}

} // namespace book
} // namespace engine
} // namespace nshogi
