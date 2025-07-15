//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_IO_BOOK_H
#define NSHOGI_ENGINE_IO_BOOK_H

#include <fstream>
#include <vector>


namespace nshogi {
namespace engine {

namespace book {

class Book;

} // namespace book

namespace io {
namespace book {

void save(const engine::book::Book& Book, std::ofstream&);
void load(engine::book::Book& Book, std::ifstream&);

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_IO_BOOK_H
