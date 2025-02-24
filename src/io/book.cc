//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "book.h"
#include <cinttypes>

namespace nshogi {
namespace engine {
namespace io {
namespace book {

void writeBookEntry(std::ofstream& Ofs, const nshogi::engine::book::BookEntry& BE) {
    core::Move16 BestMove = BE.bestMove();
    double WinRate = BE.winRate();
    double DrawRate = BE.drawRate();

    Ofs.write(BE.huffmanCode().data(),  (long)core::HuffmanCode::size());
    Ofs.write(reinterpret_cast<const char*>(&BestMove), sizeof(BestMove));
    Ofs.write(reinterpret_cast<const char*>(&WinRate), sizeof(WinRate));
    Ofs.write(reinterpret_cast<const char*>(&DrawRate), sizeof(DrawRate));
}

nshogi::engine::book::BookEntry readBookEntry(std::ifstream& Ifs) {
    char HC[32];
    core::Move16 Move;
    double WinRate;
    double DrawRate;

    Ifs.read(HC, (long)core::HuffmanCode::size());
    Ifs.read(reinterpret_cast<char*>(&Move), sizeof(Move));
    Ifs.read(reinterpret_cast<char*>(&WinRate), sizeof(WinRate));
    Ifs.read(reinterpret_cast<char*>(&DrawRate), sizeof(DrawRate));

    return nshogi::engine::book::BookEntry(HC, Move, WinRate, DrawRate);
}

std::vector<nshogi::engine::book::BookEntry> readBook(std::ifstream& Ifs) {
    Ifs.seekg(0, std::ios::beg);
    uint64_t UnitSize = 0;
    {
        auto Dummy = readBookEntry(Ifs);
        UnitSize = (uint64_t)Ifs.tellg();
    }

    Ifs.seekg(0, std::ios::end);
    std::streampos FileSize = Ifs.tellg();
    const uint64_t BookCount = (uint64_t)FileSize / UnitSize;
    std::vector<nshogi::engine::book::BookEntry> BookEntries;
    Ifs.seekg(0, std::ios::beg);
    for (uint64_t I = 0; I < BookCount; ++I) {
        auto BE = io::book::readBookEntry(Ifs);
        BookEntries.emplace_back(BE);
    }

    return BookEntries;
}

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi
