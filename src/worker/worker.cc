#include "worker.h"

namespace nshogi {
namespace engine {
namespace worker {

Worker::Worker()
    : IsRunning(false)
    , IsWaiting(false)
    , IsExiting(false)
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
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsRunning = true;
    }
    CV.notify_one();
}

void Worker::stop() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsRunning = false;
    }
    CV.notify_one();
}

void Worker::await() {
    std::unique_lock<std::mutex> Lock(Mutex);
    WaitingCV.wait(Lock, [this] {
        return !IsRunning && IsWaiting;
    });
}

void Worker::mainLoop() {
    while (true) {
        {
            std::unique_lock<std::mutex> Lock(Mutex);
            IsWaiting = true;

            WaitingCV.notify_all();
            CV.wait(Lock, [this] {
                return IsRunning || IsExiting;
            });

            IsWaiting = false;
            if (IsExiting) {
                break;
            }
        }

        while (true) {
            bool ToContinue = doTask();

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
        IsWaiting = true;
    }

    WaitingCV.notify_all();
}

} // namespace worker
} // namespace engine
} // namespace nshogi
