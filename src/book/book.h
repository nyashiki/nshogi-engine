//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOK_H
#define NSHOGI_ENGINE_BOOK_BOOK_H

#include <map>
#include <string>
#include <vector>

#include "../io/book.h"

#include <nshogi/core/state.h>
#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace book {

class Book {
 public:
    Book();

    // Note: returns copy in case `BookData` is updated after the call.
    auto nextMoves(const core::State&) const -> std::vector<core::Move16>;

    auto size() const -> std::size_t;

 private:
    std::map<std::string, std::vector<core::Move16>> BookData;

    friend auto
    io::book::load(const std::string& Path,
                   io::book::BookFormat Format) -> engine::book::Book;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOK_H
