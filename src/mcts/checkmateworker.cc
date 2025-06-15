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
#include <chrono>

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateWorker::CheckmateWorker(std::size_t Id, CheckmateQueue* CQueue, Statistics* Stat)
    : worker::Worker(true)
    , MyId(Id)
    , DfPnSolver(64)
    , PCheckmateQueue(CQueue)
    , PStat(Stat) {

    spawnThread();
}

CheckmateWorker::~CheckmateWorker() {
}

bool CheckmateWorker::doTask() {
    std::queue<std::unique_ptr<CheckmateTask>> Tasks =
        PCheckmateQueue->getAll(MyId);

    if (Tasks.empty()) {
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
        if (!Task->node()->getSolverResult().isNone()) {
            continue;
        }

        // Now, trying to solve the position.
        const auto StartTime = std::chrono::steady_clock::now();
        auto State = core::StateBuilder::newState(Task->position());
        const auto CheckmateMove = DfPnSolver.solve(&State, 10000, Task->depth());
        const auto EndTime = std::chrono::steady_clock::now();
        const uint64_t Elapsed = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                EndTime - StartTime).count();

        PStat->incrementNumSolverWorked();
        PStat->updateSolverElapsed(Elapsed);

        if (!CheckmateMove.isNone()) {
            Task->node()->setSolverResult(core::Move16(CheckmateMove));
            Task->node()->setPlyToTerminalSolved((int16_t)2048);
        } else {
            // No solver moves has been found, and mark the node
            // tried-to-solve by MoveInvalid(), which is different
            // from MoveNone().
            Task->node()->setSolverResult(core::Move16::MoveInvalid());
        }
    }

    return true;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
