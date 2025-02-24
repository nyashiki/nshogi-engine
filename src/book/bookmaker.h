//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
#define NSHOGI_ENGINE_BOOK_BOOKMAKER_H

#include <memory>
#include <set>
#include <map>
#include <cinttypes>

#include "bookentry.h"
#include "../context.h"
#include "../mcts/manager.h"

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace book {

class BookMaker {
 public:
    BookMaker(const Context* Context, std::shared_ptr<logger::Logger> Logger);

    void enumerateBookSeeds(uint64_t NumGenerates, const std::string& Path);
    void makeBookFromBookSeed(const std::string& BookSeedPath, const std::string& OutPath);
    void refineBook(const std::string& UnrefinedPath);

 private:
    BookEntry doMinMaxSearchOnBook(core::State* State, std::map<core::HuffmanCode, BookEntry>& BookEntries, std::set<core::HuffmanCode>& Fixed);
    std::unique_ptr<mcts::Manager> Manager;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
