#ifndef NSHOGI_ENGINE_MCTS_CHECKMATESEARCHER_H
#define NSHOGI_ENGINE_MCTS_CHECKMATESEARCHER_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <cinttypes>

#include "../mcts/node.h"
#include <nshogi/core/state.h>
#include <nshogi/core/statebuilder.h>

namespace nshogi {
namespace engine {
namespace mcts {

class CheckmateSearcher {
 public:
    CheckmateSearcher(int Depth_, std::size_t NumWorkers);
    ~CheckmateSearcher();

    void start();
    void stop();

    void addTask(Node* N, const core::Position& Position);

 private:
    struct Task {
     public:
        Task(Node* N_, const core::Position& P)
            : N(N_)
            , Position(P) {
        }

        Node* getNode() {
            return N;
        }

        core::Position& getPosition() {
            return Position;
        }

     private:
        Node* N;
        core::Position Position;
    };

    void mainLoop();

    const int Depth;

    std::atomic<bool> IsRunning;
    std::atomic<uint8_t> NumSearchingThreads;
    std::atomic<bool> IsExiting;
    std::condition_variable Cv;
    std::mutex Mtx;
    std::vector<std::thread> Workers;
    std::queue<Task> Tasks;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATESEARCHER_H
