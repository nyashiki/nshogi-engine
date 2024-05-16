#include "worker.h"

#include <iostream>

namespace nshogi {
namespace engine {
namespace worker {

Worker::Worker(bool LoopTask)
    : IsRunning(false)
    , IsWaiting(false)
    , IsExiting(false)
    , LoopTaskFlag(LoopTask)
    , Thread(&Worker::mainLoop, this) {
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
    std::cerr << ">>>>> START HAS CALLED. <<<<<" << std::endl;
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsRunning = true;
        IsStartNotified = false;
    }
    CV.notify_one();

    std::cerr << ">>>>> CONFIRM THREAD HAS STARTED. <<<<<" << std::endl;
    // Ensure the thread has started.
    std::unique_lock<std::mutex> Lock(Mutex);
    WaitingCV.wait(Lock, [this]() {
        return IsStartNotified;
    });

    std::cerr << ">>>>> CONFIRM THREAD HAS STARTED OK!. <<<<<" << std::endl;
}

void Worker::stop() {
    std::lock_guard<std::mutex> Lock(Mutex);
    IsRunning = false;
}

void Worker::await() {
    // Wait until the thread has stopped.
    std::unique_lock<std::mutex> Lock(Mutex);
    WaitingCV.wait(Lock, [this] {
        return !IsRunning && IsWaiting;
    });
}

bool Worker::getIsRunning() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return IsRunning;
}

void Worker::mainLoop() {
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

            CV.wait(Lock, [this] {
                return IsRunning || IsExiting;
            });

            // Notify a waiting thread which is in start().
            IsWaiting = false;
            IsToExit = IsExiting;
            IsStartNotified = true;
        }

        WaitingCV.notify_all();
        std::cerr << ">>>>> NOTIFY THREAD HAS STARTED OK!. <<<<<" << std::endl;

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
