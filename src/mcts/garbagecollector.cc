#include "garbagecollector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <iostream>

namespace nshogi {
namespace engine {
namespace mcts {

GarbageCollector::GarbageCollector(std::size_t NumWorkers) {
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

void GarbageCollector::addGarbage(std::unique_ptr<Node>&& Node) {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        Garbages.push(std::move(Node));
    }
    Cv.notify_one();
}

void GarbageCollector::mainLoop() {
    while (true) {
        std::queue<std::unique_ptr<Node>> NodesToProcess;
        {
            std::unique_lock<std::mutex> Lock(Mtx);

            Cv.wait(Lock, [&]{
                return !Garbages.empty() || ToExit;
            });

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
                NodesToProcess.push(std::move(NodeToProcess->getEdge(I)->getTargetWithOwner()));
            }
        }

        if (ToExit) {
            break;
        }
    }
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
