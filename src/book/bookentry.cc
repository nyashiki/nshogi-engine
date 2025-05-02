//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookentry.h"

namespace nshogi {
namespace engine {
namespace book {

BookEntry::BookEntry(core::HuffmanCode HC, core::Move16 BM, double WR,
                     double DR)
    : HuffmanCode(HC)
    , BestMove(BM)
    , WinRate(WR)
    , DrawRate(DR) {
}

const core::HuffmanCode& BookEntry::huffmanCode() const {
    return HuffmanCode;
}

core::Move16 BookEntry::bestMove() const {
    return BestMove;
}

double BookEntry::winRate() const {
    return WinRate;
}

double BookEntry::drawRate() const {
    return DrawRate;
}

void BookEntry::setBestMove(core::Move16 Move) {
    BestMove = Move;
}

void BookEntry::setWinRate(double Value) {
    WinRate = Value;
}

void BookEntry::setDrawRate(double Value) {
    DrawRate = Value;
}

} // namespace book
} // namespace engine
} // namespace nshogi
