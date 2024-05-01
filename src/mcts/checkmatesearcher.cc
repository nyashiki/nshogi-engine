#include "checkmatesearcher.h"

#include <iostream>

#include <nshogi/solver/dfs.h>

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateSearcher::CheckmateSearcher(int Depth_, std::size_t NumWorkers)
    : Depth(Depth_)
    , IsRunning(false)
    , NumSearchingThreads(0)
    , IsExiting(false) {

    for (std::size_t I = 0; I < NumWorkers; ++I) {
        Workers.emplace_back(&CheckmateSearcher::mainLoop, this);
    }
}

CheckmateSearcher::~CheckmateSearcher() {
    {
        std::lock_guard<std::mutex> Lk(Mtx);
        IsExiting.store(true);
    }
    Cv.notify_all();

    for (auto& Worker : Workers) {
        Worker.join();
    }
}

void CheckmateSearcher::start() {
    {
        std::lock_guard<std::mutex> Lk(Mtx);
        IsRunning.store(true);
    }
    Cv.notify_one();
}

void CheckmateSearcher::stop() {
    IsRunning.store(false);

    std::queue<Task> Dummy;
    {
        // Clear the queue.
        std::lock_guard<std::mutex> Lk(Mtx);
        Tasks.swap(Dummy);
    }

    // Spin lock until the searching thread stops.
    std::cout << "wait until issearching is false." << std::endl;
    while (NumSearchingThreads.load() > 0) {
        // Pop queue items here to decrease deconstructor's time consumption.
        if (Dummy.size() > 0) {
            Dummy.pop();
        }
    }
    std::cout << "now issearching is true." << std::endl;
}

void CheckmateSearcher::addTask(Node* N, const core::Position& Position) {
    Task&& T { N, Position };
    {
        std::lock_guard<std::mutex> Lk(Mtx);
        Tasks.push(std::move(T));
    }
    Cv.notify_one();
}

void CheckmateSearcher::mainLoop() {
    NumSearchingThreads.fetch_add(1);

    while (true) {
        std::queue<Task> Ts;

        {
            std::unique_lock<std::mutex> Lk(Mtx);

            NumSearchingThreads.fetch_sub(1);
            Cv.wait(Lk, [this]() {
                return IsExiting.load() || (IsRunning.load() && !Tasks.empty());
            });

            if (IsExiting.load()) {
                return;
            }

            NumSearchingThreads.fetch_add(1);
            Tasks.swap(Ts);
        }

        while (!Ts.empty()) {
            if (!IsRunning.load()) {
                break;
            }

            Task T = std::move(Ts.front());
            Ts.pop();

            // The solver can have ran this node already.
            if (!T.getNode()->getSolverResult().isNone()) {
                continue;
            }

            auto State = core::StateBuilder::newState(T.getPosition());
            const auto CheckmateMove = solver::dfs::solve(&State, Depth);

            if (!CheckmateMove.isNone()) {
                T.getNode()->setSolverResult(core::Move16(CheckmateMove));
                T.getNode()->setPlyToTerminalSolved((int16_t)Depth);
            } else {
                T.getNode()->setSolverResult(core::Move16::MoveInvalid());
            }
        }
    }
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
