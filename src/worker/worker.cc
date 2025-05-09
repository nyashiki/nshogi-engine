//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "worker.h"

#include <iostream>

namespace nshogi {
namespace engine {
namespace worker {

Worker::Worker(bool LoopTask)
    : IsRunning(false)
    , IsWaiting(false)
    , IsExiting(false)
    , IsStartNotified(false)
    , IsInitializationDone(false)
    , LoopTaskFlag(LoopTask) {
}

Worker::~Worker() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsExiting = true;
    }
    CV.notify_one();

    await();
    Thread.join();
}

void Worker::start() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsRunning = true;
        IsStartNotified = false;
    }
    CV.notify_one();

    // Ensure the thread has started.
    std::unique_lock<std::mutex> Lock(Mutex);
    WaitingCV.wait(Lock, [this]() { return IsStartNotified; });
}

void Worker::stop() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsRunning = false;
}

void Worker::await() {
    // Wait until the thread has stopped.
    std::unique_lock<std::mutex> Lock(Mutex);
    WaitingCV.wait(Lock, [this] { return !IsRunning && IsWaiting; });
}

void Worker::spawnThread() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Thread = std::thread(&Worker::mainLoop, this);
    }

    {
        std::unique_lock<std::mutex> Lock(MutexInitialization);
        CVInitialization.wait(Lock, [this]() { return IsInitializationDone; });
    }
}

void Worker::initializationTask() {
}

bool Worker::getIsRunning() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return IsRunning;
}

void Worker::mainLoop() {
    initializationTask();

    {
        std::lock_guard<std::mutex> Lock(MutexInitialization);
        IsInitializationDone = true;
    }
    CVInitialization.notify_one();

    while (true) {
        {
            std::lock_guard<std::mutex> Lock(Mutex);
            IsRunning = false;
            IsWaiting = true;
        }
        WaitingCV.notify_all();

        bool IsToExit = false;
        {
            std::unique_lock<std::mutex> Lock(Mutex);

            CV.wait(Lock, [this] { return IsRunning || IsExiting; });

            // Notify a waiting thread which is in start().
            IsWaiting = false;
            IsToExit = IsExiting;
            IsStartNotified = true;
        }

        WaitingCV.notify_all();

        if (IsToExit) {
            break;
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
            if (!IsRunning || IsExiting) {
                break;
            }
        }
    }

    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsRunning = false;
        IsWaiting = true;
    }
    WaitingCV.notify_all();
}

} // namespace worker
} // namespace engine
} // namespace nshogi
