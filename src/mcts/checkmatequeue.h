#ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
#define NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H

#include "node.h"

#include <memory>
#include <mutex>
#include <queue>

#include <nshogi/core/position.h>

namespace nshogi {
namespace engine {
namespace mcts {

struct CheckmateTask {
 public:
    CheckmateTask(Node* N, const core::Position& Pos)
        : TargetNode(N)
        , Position(Pos) {
    }

    Node* getNode() {
        return TargetNode;
    }

    core::Position& getPosition() {
        return Position;
    }

 private:
    Node* TargetNode;
    core::Position Position;
};

class CheckmateQueue {
 public:
    CheckmateQueue();

    void open();
    void close();
    void add(Node*, const core::Position&);
    auto getAll() -> std::queue<std::unique_ptr<CheckmateTask>>;

 private:
    bool IsOpen;
    std::mutex Mutex;
    std::queue<std::unique_ptr<CheckmateTask>> Queue;

};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEQUEUE_H
