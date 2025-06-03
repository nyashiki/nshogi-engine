//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "garbagecollector.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>

namespace nshogi {
namespace engine {
namespace mcts {

GarbageCollector::GarbageCollector(std::size_t NumWorkers,
                                   allocator::Allocator* NodeAllocator,
                                   allocator::Allocator* EdgeAllocator)
    : NA(NodeAllocator)
    , EA(EdgeAllocator) {
    if (NumWorkers <= 0) {
        std::cerr << "NumWorkers must be greater or equal than 1." << std::endl;
        std::exit(1);
    }

    ToExit = false;

    std::lock_guard<std::mutex> Lock(Mtx);
    for (std::size_t I = 0; I < NumWorkers; ++I) {
        Workers.emplace_back(&GarbageCollector::mainLoop, this);
    }
}

GarbageCollector::~GarbageCollector() {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        ToExit = true;
    }
    Cv.notify_all();

    for (auto& Worker : Workers) {
        Worker.join();
    }
}

void GarbageCollector::addGarbage(Pointer<Node>&& Node) {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        Garbages.push(std::move(Node));
    }
    Cv.notify_one();
}

void GarbageCollector::addGarbages(std::vector<Pointer<Node>>&& Nodes) {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        for (auto&& Node : Nodes) {
            Garbages.push(std::move(Node));
        }
    }
    Cv.notify_all();
}

void GarbageCollector::mainLoop() {
    while (true) {
        std::queue<Pointer<Node>> NodesToProcess;
        {
            std::unique_lock<std::mutex> Lock(Mtx);

            Cv.wait(Lock, [&] { return !Garbages.empty() || ToExit; });

            if (Garbages.empty() && ToExit) {
                break;
            }

            Garbages.swap(NodesToProcess);
        }

        while (!NodesToProcess.empty()) {
            auto NodeToProcess = std::move(NodesToProcess.front());
            NodesToProcess.pop();

            if (NodeToProcess == nullptr) {
                continue;
            }

            // To avoid stack-overflow, manually expand the children.
            const uint16_t NumChildren = NodeToProcess->getNumChildren();
            for (uint16_t I = 0; I < NumChildren; ++I) {
                NodesToProcess.push(std::move(
                    NodeToProcess->getEdge()[I].getTargetWithOwner()));
            }

            NodeToProcess->getEdge().destroy(EA, NumChildren);
            NodeToProcess.destroy(NA, 1);
        }
    }
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
