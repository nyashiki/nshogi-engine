//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "book.h"
#include "../book/book.h"

#include <fstream>
#include <memory>
#include <sstream>

#include <nshogi/core/state.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace io {
namespace book {

auto load(const std::string& Path, BookFormat Format) -> engine::book::Book {
    if (Format != BookFormat::YaneuraOu) {
        throw std::runtime_error("Unsupported book format: " +
                                 std::to_string((int)Format));
    }

    std::ifstream Ifs(Path);

    if (!Ifs) {
        throw std::runtime_error("Failed to open the book file: " + Path);
    }

    std::string Line;

    std::unique_ptr<core::State> State = nullptr;
    std::vector<core::Move16> Moves;
    uint64_t LineCount = 0;

    engine::book::Book B;

    while (std::getline(Ifs, Line)) {
        ++LineCount;

        if (Line.empty() || Line[0] == '#') {
            continue;
        }

        if (Line.substr(0, 4) == "sfen") {
            if (State != nullptr) {
                if (Moves.empty()) {
                    throw std::runtime_error(
                        "Invalid book file format: " + Path +
                        " (line: " + std::to_string(LineCount) + ")");
                }

                // Note: if the same position appears multiple times, the last
                // one is used.
                B.BookData[nshogi::io::sfen::positionToSfen(
                    State->getPosition())] = Moves;
            }

            State = std::make_unique<core::State>(
                nshogi::io::sfen::StateBuilder::newState(Line.substr(5)));
            Moves.clear();
        } else {
            if (State == nullptr) {
                throw std::runtime_error(
                    "Invalid book file format: " + Path +
                    " (line: " + std::to_string(LineCount) + ")");
            }
            std::istringstream Iss(Line);
            std::string MoveStr;
            Iss >> MoveStr;

            const auto Move =
                nshogi::io::sfen::sfenToMove32(State->getPosition(), MoveStr);
            Moves.push_back(core::Move16(Move));
        }
    }

    if (State != nullptr) {
        if (Moves.empty()) {
            throw std::runtime_error("Invalid book file format: " + Path +
                                     " (line: " + std::to_string(LineCount) +
                                     ")");
        }
        B.BookData[nshogi::io::sfen::positionToSfen(State->getPosition())] =
            Moves;
    }

    return B;
}

} // namespace book
} // namespace io
} // namespace engine
} // namespace nshogi
