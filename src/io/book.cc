//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "book.h"
#include "../book/bookentry.h"
#include <cinttypes>

namespace nshogi {
namespace engine {
namespace io {
namespace book {

void save(const engine::book::Book& Book, std::ofstream& Ofs) {
    for (const auto& [Sfen, Index] : Book.Dictionary) {
        const auto& Entry = Book.Entries[Index];

        const std::size_t Size = Sfen.size() + 1;
        Ofs.write(reinterpret_cast<const char*>(&Size), sizeof(std::size_t));
        Ofs.write(Sfen.c_str(), (long)Size);

        const uint32_t MoveValue = Entry.BestMove.value();
        Ofs.write(reinterpret_cast<const char*>(&Entry.WinRate), sizeof(double));
        Ofs.write(reinterpret_cast<const char*>(&Entry.DrawRate), sizeof(double));
        Ofs.write(reinterpret_cast<const char*>(&MoveValue), sizeof(uint32_t));
    }
}

void load(engine::book::Book& Book, std::ifstream& Ifs) {
    std::vector<char> Buffer(1024);

    Book.Dictionary.clear();
    Book.Entries.clear();

    while (true) {
        std::size_t Size = 0;
        Ifs.read(reinterpret_cast<char*>(&Size), sizeof(std::size_t));

        if (Ifs.eof()) {
            break;
        }

        Ifs.read(Buffer.data(), (long)Size);

        engine::book::BookEntry Entry;
        uint32_t MoveValue = 0;
        Ifs.read(reinterpret_cast<char*>(&Entry.WinRate), sizeof(double));
        Ifs.read(reinterpret_cast<char*>(&Entry.DrawRate), sizeof(double));
        Ifs.read(reinterpret_cast<char*>(&MoveValue), sizeof(uint32_t));
        Entry.BestMove = core::Move32::fromValue(MoveValue);

        Book.Entries.push_back(Entry);
        const std::string Sfen = std::string(Buffer.data());
        Book.Dictionary[Sfen] = Book.Entries.size() - 1;
    }
}

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi
