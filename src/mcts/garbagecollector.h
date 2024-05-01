#ifndef NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H
#define NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H

#include <cstddef>
#include <thread>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "node.h"

namespace nshogi {
namespace engine {
namespace mcts {

class GarbageCollector {
 public:
    GarbageCollector(std::size_t NumWorkers);
    ~GarbageCollector();

    void addGarbage(std::unique_ptr<Node>&& Node);

 private:
    std::mutex Mtx;
    std::condition_variable Cv;
    bool ToExit;

    void mainLoop();

    std::vector<std::thread> Workers;
    std::queue<std::unique_ptr<Node>> Garbages;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_GARBAGECOLLECTOR_H
