//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
#define NSHOGI_ENGINE_BOOK_BOOKMAKER_H

#include <cinttypes>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include "bookentry.h"
#include "../context.h"
#include "../mcts/manager.h"
#include "../contextmanager.h"

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace book {

struct Evaluation {
 public:
    Evaluation(double Win, double Draw)
        : WinRate(Win)
        , DrawRate(Draw) {
    }

    double winRate() const {
        return WinRate;
    }

    double drawRate() const {
        return DrawRate;
    }

 private:
    double WinRate;
    double DrawRate;
};

class BookMaker {
 public:
    void start(const std::string& Sfen);

 private:
    auto startThinking(
            core::State* State,
            const core::StateConfig& Config,
            const std::vector<core::Move32>& BannedMoves,
            const engine::Limit&
    ) -> std::pair<core::Move32, std::unique_ptr<mcts::ThoughtLog>>;
    void evaluate(core::State* State, const core::StateConfig& Config);
    void updateNegaMaxValue(core::State* State, const core::StateConfig& Config);
    std::optional<BookEntry> updateNegaMaxValueAllInternal(
            core::State* State,
            const core::StateConfig& Config,
            std::set<std::string>& Fixed);
    void updateNegaMaxValueAll(const core::StateConfig& Config);
    void executeOneIteration(core::State* State, const core::StateConfig& Config);
    std::vector<core::Move32> getPV(core::State* State, const core::StateConfig& Config);

    std::unique_ptr<mcts::Manager> Manager;
    Book MyBook;

    ContextManager CManager;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
