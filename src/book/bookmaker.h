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

#include <cstddef>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "node.h"
#include "../mcts/manager.h"
#include "../contextmanager.h"

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/solver/dfpn.h>

namespace nshogi {
namespace engine {

namespace book {

class BookMaker;

} // namespace book

namespace io {
namespace book {

void save(const engine::book::BookMaker& Maker, std::ofstream& IndexOfs, std::ofstream& DataOfs);
void load(engine::book::BookMaker* Maker, std::ifstream& IndexIfs, std::ifstream& DataIfs);

} // namespace book
} // namespace io

namespace book {

class BookMaker {
 public:
    explicit BookMaker();

    void start(const core::State& RootState, uint64_t NumSimulations);

 private:
    void prepareMCTSManager();
    void prepareRoot(const core::State& RootState);

    auto collectOneLeaf(core::State*) -> std::tuple<std::vector<std::pair<NodeIndex, std::size_t>>, float, float>;
    void expandAndEvaluate(core::State*, NodeIndex);
    void evaluate(const core::State*, NodeIndex);
    void backpropagate(const std::vector<std::pair<NodeIndex, std::size_t>>& Trajectory, float WinRate, float DrawRate);

    auto computeUCBMaxChild(core::State*, NodeIndex) -> std::pair<std::size_t, core::Move32>;
    void debugOutput() const;
    auto currentPV(NodeIndex) const -> std::vector<core::Move32>;

    std::unordered_map<std::string, NodeIndex> NodeIndices;
    std::vector<Node> Nodes;

    solver::dfpn::Solver Solver;
    ContextManager CManager;
    core::StateConfig StateConfig;
    std::unique_ptr<mcts::Manager> MCTSManager;

 friend void engine::io::book::save(const BookMaker& Maker, std::ofstream&, std::ofstream&);
 friend void engine::io::book::load(BookMaker* Maker, std::ifstream&, std::ifstream&);
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
