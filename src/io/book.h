//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_IO_BOOK_H
#define NSHOGI_ENGINE_IO_BOOK_H

#include <string>

namespace nshogi {
namespace engine {

namespace book {

class Book;

} // namespace book

namespace io {
namespace book {

enum class BookFormat {
    YaneuraOu,
};

auto load(const std::string& Path, BookFormat Format) -> engine::book::Book;

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_IO_BOOK_H
