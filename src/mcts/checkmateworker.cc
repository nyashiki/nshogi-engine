//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "checkmateworker.h"

#include <nshogi/core/statebuilder.h>
#include <nshogi/solver/dfs.h>

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateWorker::CheckmateWorker(CheckmateQueue* CQueue)
    : worker::Worker(true)
    , SolverDepth(5)
    , PCheckmateQueue(CQueue) {

    spawnThread();
}

CheckmateWorker::~CheckmateWorker() {
}

bool CheckmateWorker::doTask() {
    std::queue<std::unique_ptr<CheckmateTask>> Tasks =
        PCheckmateQueue->getAll();

    if (Tasks.empty()) {
        std::this_thread::yield();
        return false;
    }

    while (!Tasks.empty()) {
        std::unique_ptr<CheckmateTask> Task = std::move(Tasks.front());
        Tasks.pop();

        if (!isRunning()) {
            // This solver has been told to stop.
            break;
        }

        // This node has been tried to be solved by the solvers.
        if (!Task->getNode()->getSolverResult().isNone()) {
            continue;
        }

        // Now, trying to solve the position.
        auto State = core::StateBuilder::newState(Task->getPosition());
        const auto CheckmateMove = solver::dfs::solve(&State, SolverDepth);

        if (!CheckmateMove.isNone()) {
            Task->getNode()->setSolverResult(core::Move16(CheckmateMove));
            Task->getNode()->setPlyToTerminalSolved((int16_t)SolverDepth);
        } else {
            // No solver moves has been found, and mark the node
            // tried-to-solve by MoveInvalid(), which is different
            // from MoveNone().
            Task->getNode()->setSolverResult(core::Move16::MoveInvalid());
        }
    }

    return true;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
