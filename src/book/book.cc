//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "book.h"

#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace book {

Book::Book() {
}

auto Book::nextMoves(const core::State& State) const
    -> std::vector<core::Move16> {
    const std::string Key = nshogi::io::sfen::positionToSfen(State.getPosition());
    const auto It = BookData.find(Key);

    if (It != BookData.end()) {
        return It->second;
    }

    return { };
}

} // namespace book
} // namespace engine
} // namespace nshogi
