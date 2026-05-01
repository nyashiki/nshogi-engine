//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H
#define NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H

#include "checkmatetask.h"
#include "node.h"
#include "pointer.h"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nshogi {
namespace engine {
namespace mcts {

class GarbageCollector {
 public:
    GarbageCollector(std::size_t NumWorkers,
                     allocator::Allocator* NodeAllocator,
                     allocator::Allocator* EdgeAllocator);
    ~GarbageCollector();

    void addGarbage(Pointer<Node>&& Node);
    void addGarbages(std::vector<Pointer<Node>>&& Nodes);
    void addCheckmateGarbage(std::queue<std::unique_ptr<CheckmateTask>>&& Tasks);

 private:
    std::mutex Mtx;
    std::condition_variable Cv;
    bool ToExit;

    void mainLoop();

    std::vector<std::thread> Workers;
    std::queue<Pointer<Node>> Garbages;
    std::queue<std::queue<std::unique_ptr<CheckmateTask>>> CheckmateGarbages;

    allocator::Allocator* NA;
    allocator::Allocator* EA;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H
