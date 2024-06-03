#ifndef NSHOGI_ENGINE_WORKER_WORKER_H
#define NSHOGI_ENGINE_WORKER_WORKER_H

#include <condition_variable>
#include <mutex>
#include <thread>

namespace nshogi {
namespace engine {
namespace worker {

class Worker {
 public:
    Worker(bool LoopTask);
    virtual ~Worker();

    void start();
    void stop();
    void await();

 protected:
    // spawnThread() must be called exactly once in
    // the constructor of a child class.
    void spawnThread();

    virtual void initializationTask();
    virtual bool doTask() = 0;
    bool getIsRunning();

    bool IsRunning;
    bool IsWaiting;
    bool IsExiting;
    bool IsStartNotified;
    bool IsInitializationDone;

 private:
    void mainLoop();

    const bool LoopTaskFlag;

    std::thread Thread;
    std::mutex Mutex;
    std::mutex MutexInitialization;
    std::condition_variable CV;
    std::condition_variable CVInitialization;
    std::condition_variable WaitingCV;
};

} // namespace worker
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_WORKER_WORKER_H
