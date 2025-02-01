//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

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
#include "commands/stop.h"
#include "../mcts/manager.h"
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

    void pushCommand(std::shared_ptr<ICommand> Command, bool blocking = false);

    const Context* getContext() const;

 private:
    void mainLoop();
    void executeCommand(std::shared_ptr<ICommand> Command);

    void executeCommand(const commands::Noop* Command);
    void executeCommand(const commands::IConfig* Command);
    void executeCommand(const commands::GetReady* Command);
    void executeCommand(const commands::SetPosition* Command);
    void executeCommand(const commands::Think* Command);
    void executeCommand(const commands::Stop* Command);

    void setConfig(const commands::BoolConfig* Config);
    void setConfig(const commands::IntegerConfig* Config);
    void setConfig(const commands::DoubleConfig* Config);
    void setConfig(const commands::StringConfig* Config);

    std::deque<std::shared_ptr<ICommand>> CommandQueue;
    std::thread Worker;

    bool IsExiting;
    std::mutex Mtx;
    std::condition_variable CV;

    ContextManager CManager;

    std::unique_ptr<core::State> State;
    std::unique_ptr<core::StateConfig> StateConfig;

    std::unique_ptr<mcts::Manager> Manager;

    std::shared_ptr<logger::Logger> PLogger;
};

} // namespace command
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_COMMAND_EXECUTOR_H
