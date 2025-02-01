//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_TREE_H
#define NSHOGI_ENGINE_MCTS_TREE_H


#include "node.h"
#include "edge.h"
#include "garbagecollector.h"
#include "../logger/logger.h"

#include <memory>
#include <nshogi/core/state.h>


namespace nshogi {
namespace engine {
namespace mcts {

class Tree {
 public:
    Tree(GarbageCollector* GCollector, allocator::Allocator* NodeAllocator, logger::Logger* Logger);
    ~Tree();

    Node* updateRoot(const nshogi::core::State& State, bool ReUse = true);

    Node* getRoot();
    const core::State* getRootState() const;

 private:
    Node* createNewRoot(const nshogi::core::State& State);

    Pointer<Node> Root;
    std::unique_ptr<core::State> RootState;
    GarbageCollector* GC;

    allocator::Allocator* NA;
    logger::Logger* PLogger;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_TREE_H
