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

namespace nshogi {
namespace engine {
namespace book {

struct BookEntry {
 private:
    core::HuffmanCode HuffmanCode;
    core::Move16 BestMove;
    double WinRate;
    double DrawRate;
};

} // namespace book
} // namespace engine
} // namespsace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKENTRY_H
