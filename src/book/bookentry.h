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

#include <nshogi/core/huffman.h>
#include <nshogi/core/types.h>

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
        , DrawRate(0.0) {
    }

    double WinRate;
    double DrawRate;
    core::Move32 BestMove;
};

class Book {
 public:
    Book() {
    }

    void update(const std::string& Sfen, const BookEntry& Entry) {
        if (Dictionary.contains(Sfen)) {
            const std::size_t Index = Dictionary[Sfen];
            Entries[Index] = Entry;
        } else {
            Entries.push_back(Entry);
            Dictionary.emplace(Sfen, Entries.size() - 1);
        }
    }

    const BookEntry* get(const std::string& Sfen) const {
        if (Dictionary.contains(Sfen)) {
            return &Entries.at(Dictionary.at(Sfen));
        } else {
            return nullptr;
        }
    }

 private:
    std::map<std::string, std::size_t> Dictionary;
    std::vector<BookEntry> Entries;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKENTRY_H
