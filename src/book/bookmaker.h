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
#include <string>
#include <vector>
#include <iostream>

#include "node.h"

#include <nshogi/solver/dfpn.h>

namespace nshogi {
namespace engine {
namespace book {

class BookMaker {
 public:
    explicit BookMaker();

    void start(const core::State& RootState, uint64_t NumSimulations);

 private:
    void prepareRoot(const core::State& RootState);
    auto collectOneLeaf(core::State*) -> NodeIndex;
    void expandAndEvaluate(core::State*, NodeIndex);
    void evaluate(NodeIndex);
    void backpropagate(NodeIndex, float WinRate, float DrawRate);

    auto computeUCBMaxChild(core::State*, NodeIndex) -> std::pair<NodeIndex, core::Move32>;
    void debugOutput() const;
    auto currentPV(NodeIndex) const -> std::vector<core::Move32>;

    std::unordered_map<std::string, NodeIndex> NodeIndices;
    std::vector<Node> Nodes;

    solver::dfpn::Solver Solver;
};

} // namespace book
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BOOK_BOOKMAKER_H
