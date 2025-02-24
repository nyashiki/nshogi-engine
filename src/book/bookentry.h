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

#include <nshogi/core/types.h>
#include <nshogi/core/huffman.h>

namespace nshogi {
namespace engine {
namespace book {

struct BookEntry {
 public:
    BookEntry(core::HuffmanCode, core::Move16, double, double);

    const core::HuffmanCode& huffmanCode() const;
    core::Move16 bestMove() const;
    double winRate() const;
    double drawRate() const;

    void setBestMove(core::Move16 Move);
    void setWinRate(double);
    void setDrawRate(double);

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
