//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "checkmateworker.h"

#include <chrono>
#include <nshogi/core/statebuilder.h>

namespace nshogi {
namespace engine {
namespace mcts {

CheckmateWorker::CheckmateWorker(CheckmateQueue* CQueue, Statistics* Stat)
    : worker::Worker(true)
    , DfPnSolver(64)
    , PCheckmateQueue(CQueue)
    , LatestGeneration(0)
    , PStat(Stat) {

    spawnThread();
}

CheckmateWorker::~CheckmateWorker() {
}

bool CheckmateWorker::doTask() {
    std::unique_ptr<CheckmateTask> Task = PCheckmateQueue->get();

    if (Task == nullptr) {
        return false;
    }

    if (Task->generation() != LatestGeneration) {
        PCheckmateQueue->lock();
        LatestGeneration = PCheckmateQueue->_generation();
        PCheckmateQueue->unlock();
        if (Task->generation() != LatestGeneration) {
            return false;
        }
    }

    // This node has been tried to be solved by the solvers.
    if (!Task->node()->getSolverResult().isNone()) {
        return false;
    }

    // Now, trying to solve the position.
    const auto StartTime = std::chrono::steady_clock::now();
    auto State = core::StateBuilder::newState(Task->position());
    const auto CheckmateMove = DfPnSolver.solve(&State, 1000, Task->depth());
    const auto EndTime = std::chrono::steady_clock::now();
    const uint64_t Elapsed =
        (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
            EndTime - StartTime)
            .count();

    PStat->incrementNumSolverWorked();
    PStat->updateSolverElapsed(Elapsed);

    PCheckmateQueue->lock();
    LatestGeneration = PCheckmateQueue->_generation();
    if (Task->generation() == LatestGeneration) {
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
    PCheckmateQueue->unlock();

    return false;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
