//
// Copyright (c) 2025-2026 @nyashiki
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

    if (Task->generation() != LatestGeneration.load(std::memory_order_acquire)) {
        return false;
    }

    // This node has been tried to be solved by the solvers.
    if (!Task->node()->getSolverResult().isNone()) {
        return false;
    }

    // Now, trying to solve the position.
    const auto StartTime = std::chrono::steady_clock::now();
    auto State = core::StateBuilder::newState(Task->position());
    const auto CheckmateSequence =
        DfPnSolver.solveWithPV(&State, 1000, Task->depth());
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
        if (!CheckmateSequence.empty()) {
            Task->node()->setSolverResult(core::Move16(CheckmateSequence[0]));
            Task->node()->setPlyToTerminalSolved(
                (int16_t)CheckmateSequence.size());
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

void CheckmateWorker::setGeneration(uint64_t Gen) noexcept {
    LatestGeneration.store(Gen, std::memory_order_release);
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
