//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_WORKER_WORKER_H
#define NSHOGI_ENGINE_WORKER_WORKER_H

#include <condition_variable>
#include <mutex>
#include <thread>
#include <stop_token>

namespace nshogi {
namespace engine {
namespace worker {

enum class WorkerState {
    Uninitialized,
    Idle,
    Running,
    Exiting,
    Exit,
};

class Worker {
 public:
    Worker(bool LoopTask);
    virtual ~Worker();

    virtual void start();
    virtual void stop();
    virtual void await();

    bool isRunning();

 protected:
    // spawnThread() must be called exactly once in
    // the constructor of a child class.
    void spawnThread();

    virtual void initializationTask();
    virtual bool doTask() = 0;

 private:
    static constexpr uint64_t STREAK_RUN_PERIOD = 4;

    void mainLoop();

    const bool LoopTaskFlag;
    WorkerState WState;

    std::thread Thread;
    std::mutex Mutex;
    std::condition_variable TaskCV;
    std::condition_variable InitializationCV;
    std::condition_variable AwaitCV;
    std::stop_source StopSource;
};

} // namespace worker
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_WORKER_WORKER_H
