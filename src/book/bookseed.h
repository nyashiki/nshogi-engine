//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOKSEED_H
#define NSHOGI_ENGINE_BOOK_BOOKSEED_H

#include <nshogi/core/state.h>
#include <nshogi/core/huffman.h>

namespace nshogi {
namespace engine {
namespace book {

struct BookSeed {
 public:
    BookSeed(const core::State& State, double LogProbability);
    BookSeed(const core::State& State, double LogProbability, const core::HuffmanCode& ParentHuffman);

    const core::HuffmanCode& huffmanCode() const;
    double logProbability() const;
    bool hasParent() const;
    const core::HuffmanCode& parentCode() const;

 private:
    core::HuffmanCode HuffmanCode;
    double LP;
    bool HasParent;
    core::HuffmanCode Parent;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKSEED_H
