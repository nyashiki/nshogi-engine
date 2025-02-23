//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "bookseed.h"

namespace nshogi {
namespace engine {
namespace book {

BookSeed::BookSeed(const core::State& State, double LogProbability)
    : HuffmanCode(core::HuffmanCode::encode(State.getPosition()))
    , LP(LogProbability)
    , HasParent(false)
    , Parent(0, 0, 0, 0) {
}

BookSeed::BookSeed(const core::State& State, double LogProbability, const core::HuffmanCode& ParentHuffman)
    : HuffmanCode(core::HuffmanCode::encode(State.getPosition()))
    , LP(LogProbability)
    , HasParent(true)
    , Parent(ParentHuffman) {
}

const core::HuffmanCode& BookSeed::huffmanCode() const {
    return HuffmanCode;
}

double BookSeed::logProbability() const {
    return LP;
}

bool BookSeed::hasParent() const {
    return HasParent;
}

const core::HuffmanCode& BookSeed::parentCode() const {
    return Parent;
}

} // namespace book
} // namespace engine
} // namespace nshogi
