//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "worker.h"

#include <cassert>

namespace nshogi {
namespace engine {
namespace worker {

Worker::Worker(bool LoopTask)
    : LoopTaskFlag(LoopTask)
    , WState(WorkerState::Uninitialized) {
}

Worker::~Worker() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        assert(WState == WorkerState::Idle);
        WState = WorkerState::Exiting;
    }
    TaskCV.notify_one();

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        AwaitCV.wait(Lock, [this] { return WState == WorkerState::Exit; });
    }

    Thread.join();
}

void Worker::start() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        assert(WState == WorkerState::Idle);
        WState = WorkerState::Running;
        StopSource = std::stop_source();
    }
    TaskCV.notify_one();
}

void Worker::stop() {
    StopSource.request_stop();
}

void Worker::await() {
    // Wait until the thread has stopped.
    std::unique_lock<std::mutex> Lock(Mutex);
    AwaitCV.wait(Lock, [this] { return WState == WorkerState::Idle; });
}

void Worker::spawnThread() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Thread = std::thread(&Worker::mainLoop, this);
    }

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        InitializationCV.wait(
            Lock, [this]() { return WState != WorkerState::Uninitialized; });
    }
}

void Worker::initializationTask() {
}

bool Worker::isRunning() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return WState == WorkerState::Running;
}

void Worker::mainLoop() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        initializationTask();
        WState = WorkerState::Idle;
    }
    InitializationCV.notify_one();

    while (true) {
        {
            std::lock_guard<std::mutex> Lock(Mutex);
            WState = WorkerState::Idle;
            AwaitCV.notify_all();
        }

        {
            std::unique_lock<std::mutex> Lock(Mutex);

            TaskCV.wait(Lock, [this] {
                return WState == WorkerState::Running ||
                       WState == WorkerState::Exiting;
            });

            if (WState == WorkerState::Exiting) {
                break;
            }
        }

        uint64_t StreakRun = 0;

        std::stop_token StopToken;
        {
            std::lock_guard<std::mutex> Lock(Mutex);
            StopToken = StopSource.get_token();
        }

        while (true) {
            bool ToContinue = doTask();

            if (!LoopTaskFlag) {
                break;
            }

            if (ToContinue) {
                continue;
            }

            ++StreakRun;
            if (StreakRun == STREAK_RUN_PERIOD) {
                if (StopToken.stop_requested()) {
                    break;
                }
                StreakRun = 0;
            }
        }
    }

    {
        std::lock_guard<std::mutex> Lock(Mutex);
        WState = WorkerState::Exit;
    }
    AwaitCV.notify_all();
}

} // namespace worker
} // namespace engine
} // namespace nshogi
