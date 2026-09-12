//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../worker/worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace {

class TaskProbe : public nshogi::engine::worker::Worker {
 public:
    TaskProbe()
        : Worker(false) {
        spawnThread();
    }

    // Only read after await(), which synchronizes with task completion.
    int Runs = 0;

 protected:
    bool doTask() override {
        ++Runs;
        return false;
    }
};

TEST(WorkerLifecycle, ImmediateDestructionDoesNotLoseExitRequest) {
    for (int I = 0; I < 256; ++I) {
        TaskProbe Worker;
    }
}

TEST(WorkerLifecycle, ImmediateStartAndRepeatedTasksAreNotLost) {
    for (int I = 0; I < 256; ++I) {
        TaskProbe Worker;
        for (int Run = 1; Run <= 4; ++Run) {
            Worker.start();
            Worker.await();
            EXPECT_EQ(Worker.Runs, Run);
        }
    }
}

TEST(WorkerLifecycle, StopCanOverlapTheNextStart) {
    TaskProbe Worker;
    std::atomic<bool> Done{false};
    std::thread Stopper([&]() {
        while (!Done.load(std::memory_order_relaxed)) {
            Worker.stop();
            std::this_thread::yield();
        }
    });
    // A non-looping task still completes when stopped. This exercises the
    // stop_source lifetime across restarts, independently of MCTS scheduling.
    for (int I = 0; I < 256; ++I) {
        Worker.start();
        Worker.await();
    }
    Done.store(true, std::memory_order_relaxed);
    Stopper.join();
    EXPECT_EQ(Worker.Runs, 256);
}

} // namespace
