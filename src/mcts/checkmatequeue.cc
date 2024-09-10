#include "checkmatequeue.h"

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateQueue::CheckmateQueue()
    : IsOpen(false) {
}

void CheckmateQueue::open() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsOpen = true;
}

void CheckmateQueue::close() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsOpen = false;
}

void CheckmateQueue::add(Node* N, const core::Position& Position) {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (IsOpen) {
        Queue.emplace(std::make_unique<CheckmateTask>(N, Position));
    }
}

auto CheckmateQueue::getAll() -> std::queue<std::unique_ptr<CheckmateTask>> {
    std::queue<std::unique_ptr<CheckmateTask>> Q;

    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Queue.swap(Q);
    }

    return Q;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
