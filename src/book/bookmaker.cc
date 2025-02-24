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

        std::cout << ::nshogi::io::sfen::stateToSfen(State) << std::endl;
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

    uint64_t UnitSize = 0;
    {
        BookSeed Dummy = readSeed(Ifs);
        UnitSize = (uint64_t)Ifs.tellg();
    }

    Ifs.seekg(0, std::ios::end);
    std::streampos FileSize = Ifs.tellg();
    const uint64_t SeedCount = (uint64_t)FileSize / UnitSize;

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
            io::book::writeBookEntry(Ofs, BE);

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

    std::cout << "SeedCount: " << SeedCount << std::endl;
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

    std::set<core::HuffmanCode> Fixed;
    for (const auto& [Huffman, Entry] : BookEntries) {
        const auto Position = core::HuffmanCode::decode(Huffman);
        auto State = core::StateBuilder::newState(Position);
        doMinMaxSearchOnBook(&State, BookEntries, Fixed);
        std::printf("\rProgress: %.2f", (double)Fixed.size() / (double)BookEntries.size() * 100.0);
        std::cout << std::flush;

        if (Fixed.size() == BookEntries.size()) {
            break;
        }
    }
    std::cout << std::endl;

    std::cout << "BookEntries.size(): " << BookEntries.size() << std::endl;
    for (const auto& [Huffman, Entry] : BookEntries) {
        io::book::writeBookEntry(Ofs, Entry);
    }
}

BookEntry BookMaker::doMinMaxSearchOnBook(core::State* State, std::map<core::HuffmanCode, BookEntry>& BookEntries, std::set<core::HuffmanCode>& Fixed) {
    BookEntry& ThisEntry = BookEntries.find(core::HuffmanCode::encode(State->getPosition()))->second;

    if (Fixed.contains(ThisEntry.huffmanCode())) {
        return ThisEntry;
    }

    core::RepetitionStatus RS = State->getRepetitionStatus();
    if (RS != core::RepetitionStatus::NoRepetition) {
        return ThisEntry;
    }

    auto NextMoves = core::MoveGenerator::generateLegalMoves(*State);
    const BookEntry* BestEntry = nullptr;
    core::Move32 BestMove = core::Move32::MoveNone();
    double BestWinRate = -1.0;

    for (const auto& Move : NextMoves) {
        State->doMove(Move);

        if (BookEntries.find(core::HuffmanCode::encode(State->getPosition())) == BookEntries.end()) {
            State->undoMove();
            continue;
        }

        const BookEntry& BE = doMinMaxSearchOnBook(State, BookEntries, Fixed);
        State->undoMove();

        double ChildWinRate = 1.0 - BE.winRate();
        if (ChildWinRate > BestWinRate) {
            BestWinRate = ChildWinRate;
            BestMove = Move;
            BestEntry = &BE;
        }
    }

    if (core::Move16(BestMove) == ThisEntry.bestMove()) {
        ThisEntry.setWinRate(BestEntry->winRate());
        ThisEntry.setDrawRate(BestEntry->drawRate());
    } else if (BestWinRate > ThisEntry.winRate()) {
        ThisEntry.setBestMove(BestMove);
        ThisEntry.setWinRate(BestWinRate);
        ThisEntry.setDrawRate(BestEntry->drawRate());
    }
    Fixed.emplace(ThisEntry.huffmanCode());

    return ThisEntry;
}

} // namespace book
} // namespace engine
} // namespace nshogi
