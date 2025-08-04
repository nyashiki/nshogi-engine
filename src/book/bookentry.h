//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOKENTRY_H
#define NSHOGI_ENGINE_BOOK_BOOKENTRY_H

#include "../io/book.h"

#include <nshogi/core/huffman.h>
#include <nshogi/core/types.h>

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace nshogi {
namespace engine {
namespace book {

struct BookEntry {
 public:
    BookEntry()
        : WinRate(0.0)
        , DrawRate(0.0)
        , BestMove(core::Move32::MoveNone()) {
    }

    double WinRate;
    double DrawRate;
    core::Move32 BestMove;
};

class Book {
 public:
    Book() {
    }

    const BookEntry* update(const std::string& Sfen, const BookEntry& Entry) {
        if (Dictionary.contains(Sfen)) {
            const std::size_t Index = Dictionary[Sfen];
            Entries[Index] = Entry;
            return &Entries[Index];
        } else {
            Entries.push_back(Entry);
            Dictionary.emplace(Sfen, Entries.size() - 1);
            return &Entries.back();
        }
    }

    const BookEntry* get(const std::string& Sfen) const {
        if (Dictionary.contains(Sfen)) {
            return &Entries.at(Dictionary.at(Sfen));
        } else {
            return nullptr;
        }
    }

    std::size_t size() const {
        return Entries.size();
    }

    const std::map<std::string, std::size_t>& dictionary() const {
        return Dictionary;
    }

 private:
    std::map<std::string, std::size_t> Dictionary;
    std::vector<BookEntry> Entries;

 friend void nshogi::engine::io::book::save(const Book&, std::ofstream&, nshogi::engine::io::book::Format);
 friend void nshogi::engine::io::book::load(Book&, std::ifstream&);
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKENTRY_H
