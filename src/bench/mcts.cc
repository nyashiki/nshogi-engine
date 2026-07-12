//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "mcts.h"

#include "../contextmanager.h"
#include "../limit.h"
#include "../mcts/manager.h"
#include "../protocol/usilogger.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <nshogi/core/state.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace bench {

void benchMCTS(uint64_t DurationSeconds, std::size_t BatchSize,
               std::size_t NumGPUs, std::size_t NumThreadsPerGPU,
               std::size_t NumCheckmateSearchers, std::size_t EvalCacheMB) {
    // Setup the context.
    const std::size_t AvailableMemoryGB = 8UL;
    std::cout << "Warning: this method consumes " << AvailableMemoryGB
              << " GB memory." << std::endl;

    ContextManager CManager;
    CManager.setAvailableMemoryMB(AvailableMemoryGB * 1024);
    CManager.setEvalCacheMemoryMB(EvalCacheMB);
    CManager.setBatchSize(BatchSize);
    CManager.setNumGPUs(NumGPUs);
    CManager.setNumEvaluationThreadsPerGPU(NumThreadsPerGPU);
    CManager.setNumCheckmateSearchThreads(NumCheckmateSearchers);
    CManager.setIsPonderingEnabled(false);

    // Setup MCTS.
    std::cout << "Setting MCTS." << std::endl;
    std::cout << "    - NumGPUs: " << NumGPUs << std::endl;
    std::cout << "    - NumThreadsPerGPU: " << NumThreadsPerGPU << std::endl;
    std::cout << "    - NumCheckmateSearchers: " << NumCheckmateSearchers
              << std::endl;
    std::cout << "    - BatchSize: " << BatchSize << std::endl;
    auto Logger = std::make_shared<protocol::usi::USILogger>();
    mcts::Manager Manager(CManager.getContext(), Logger);

    // Setup a state.
    std::cout << "Setting a state." << std::endl;
    auto State = core::StateBuilder::getInitialState();
    auto Config = core::StateConfig();
    std::cout << "    - State: " << nshogi::io::sfen::stateToSfen(State)
              << std::endl;

    // Start thinking (without blocking).
    std::cout << "MCTS has started for " << DurationSeconds << " seconds."
              << std::endl;
    std::mutex Mtx;
    std::condition_variable CV;
    bool BestMoveArrived = false;
    core::Move32 BestMove = core::Move32::MoveNone();
    Manager.thinkNextMove(State, Config, NoLimit, [&](core::Move32 Move) {
        std::lock_guard<std::mutex> Lock(Mtx);
        BestMove = Move;
        BestMoveArrived = true;
        CV.notify_one();
    });

    std::this_thread::sleep_for(std::chrono::seconds(DurationSeconds));
    Manager.interrupt();

    // Wait until the search actually stops and reports its bestmove
    // (the statistics are printed by the manager at this point).
    {
        std::unique_lock<std::mutex> Lock(Mtx);
        CV.wait(Lock, [&]() { return BestMoveArrived; });
    }

    std::cout << "MCTS done. (bestmove: "
              << nshogi::io::sfen::move32ToSfen(BestMove) << ")" << std::endl;
}

} // namespace bench
} // namespace engine
} // namespace nshogi
