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

class BookMaker;

} // namespace book

namespace io {
namespace book {

// enum class Format {
//     NShogi,
//     YaneuraOu,
// };
//
// void save(const engine::book::Book& Book, std::ofstream&,
//           Format = Format::NShogi);
// void load(engine::book::Book& Book, std::ifstream&);

void save(const engine::book::BookMaker& Maker, std::ofstream& IndexOfs, std::ofstream& DataOfs);
void load(engine::book::BookMaker* Maker, std::ifstream& IndexIfs, std::ifstream& DataIfs);

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_IO_BOOK_H
