//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOK_H
#define NSHOGI_ENGINE_BOOK_BOOK_H

#include <string>

#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace book {

struct BookEntry {
 public:
    std::string Sfen;
    core::Move32 Move;
    float WinRate;
    float DrawRate;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // NSHOGI_ENGINE_BOOK_BOOK_H
