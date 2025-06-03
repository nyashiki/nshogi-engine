//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "worker.h"

namespace nshogi {
namespace engine {
namespace worker {

Worker::Worker(bool LoopTask)
    : IsStartNotified(false)
    , LoopTaskFlag(LoopTask)
    , WState(WorkerState::Uninitialized) {
}

Worker::~Worker() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        WState = WorkerState::Exiting;
    }
    TaskCV.notify_one();

    await();
    Thread.join();
}

void Worker::start() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        WState = WorkerState::Running;
        IsStartNotified = false;
    }
    TaskCV.notify_one();

    // Ensure the thread has started.
    std::unique_lock<std::mutex> Lock(Mutex);
    StartCV.wait(Lock, [this]() { return IsStartNotified; });
}

void Worker::stop() {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (WState == WorkerState::Running) {
        WState = WorkerState::Stopping;
    }
}

void Worker::await() {
    // Wait until the thread has stopped.
    std::unique_lock<std::mutex> Lock(Mutex);
    AwaitCV.wait(Lock, [this] {
            return WState == WorkerState::Idle ||
            WState == WorkerState::Exit;
            });
}

void Worker::spawnThread() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Thread = std::thread(&Worker::mainLoop, this);
    }

    {
        std::unique_lock<std::mutex> Lock(Mutex);
        InitializationCV.wait(Lock, [this]() { return WState != WorkerState::Uninitialized; });
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
            std::unique_lock<std::mutex> Lock(Mutex);
            WState = WorkerState::Idle;
            AwaitCV.notify_all();

            TaskCV.wait(Lock, [this] {
                    return WState == WorkerState::Running ||
                    WState == WorkerState::Exiting;
                    });

            // Notify a waiting thread which is in start().
            IsStartNotified = true;

            if (WState == WorkerState::Running) {
                StartCV.notify_all();
            } else if (WState == WorkerState::Exiting) {
                AwaitCV.notify_all();
                break;
            }
        }

        while (true) {
            bool ToContinue = doTask();

            if (!LoopTaskFlag) {
                break;
            }

            if (ToContinue) {
                continue;
            }

            std::lock_guard<std::mutex> Lock(Mutex);
            if (WState == WorkerState::Stopping) {
                break;
            }
        }
    }

    std::lock_guard<std::mutex> Lock(Mutex);
    WState = WorkerState::Exit;
    AwaitCV.notify_all();
}

} // namespace worker
} // namespace engine
} // namespace nshogi
