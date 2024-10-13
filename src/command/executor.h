#ifndef NSHOGI_ENGINE_COMMAND_EXECUTOR_H
#define NSHOGI_ENGINE_COMMAND_EXECUTOR_H

#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>

#include "command.h"
#include "commands/noop.h"
#include "commands/config.h"
#include "commands/getready.h"
#include "commands/setposition.h"
#include "commands/think.h"
#include "../logger/logger.h"
#include "../contextmanager.h"

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace command {

class Executor {
 public:
    Executor(std::shared_ptr<logger::Logger> Logger);
    ~Executor();

    void pushCommand(std::unique_ptr<ICommand>&& Command);

    const Context* getContext() const;

 private:
    void mainLoop();
    void executeCommand(std::unique_ptr<ICommand>&& Command);

    void executeCommand(const commands::Noop* Command);
    void executeCommand(std::unique_ptr<commands::IConfig>&& Command);
    void executeCommand(const commands::GetReady* Command);
    void executeCommand(const commands::SetPosition* Command);
    void executeCommand(const commands::Think* Command);

    void setConfig(const commands::BoolConfig* Config);
    void setConfig(const commands::IntegerConfig* Config);
    void setConfig(const commands::DoubleConfig* Config);
    void setConfig(const commands::StringConfig* Config);

    std::deque<std::unique_ptr<ICommand>> CommandQueue;
    std::thread Worker;

    bool IsExiting;
    std::mutex Mtx;
    std::condition_variable CV;

    ContextManager CManager;
    std::queue<std::unique_ptr<commands::IConfig>> Configs;

    std::unique_ptr<core::State> State;

    std::shared_ptr<logger::Logger> pLogger;
};

} // namespace command
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_COMMAND_EXECUTOR_H
