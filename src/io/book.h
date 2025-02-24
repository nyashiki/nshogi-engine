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

#include "../book/bookentry.h"
#include <fstream>
#include <vector>

namespace nshogi {
namespace engine {
namespace io {
namespace book {

void writeBookEntry(std::ofstream& Ofs, const nshogi::engine::book::BookEntry& BE);
nshogi::engine::book::BookEntry readBookEntry(std::ifstream& Ifs);
std::vector<nshogi::engine::book::BookEntry> readBook(std::ifstream& Ifs);


} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_IO_BOOK_H
