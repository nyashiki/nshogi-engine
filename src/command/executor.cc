#include "executor.h"

#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace command {

Executor::Executor(std::shared_ptr<logger::Logger> Logger)
    : Worker(&Executor::mainLoop, this)
    , IsExiting(false)
    , pLogger(std::move(Logger)) {
}

Executor::~Executor() {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        IsExiting = true;
    }

    CV.notify_one();
    Worker.join();
}

void Executor::pushCommand(std::unique_ptr<ICommand>&& Command) {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        CommandQueue.push_back(std::move(Command));
    }

    CV.notify_one();
}

const Context* Executor::getContext() const {
    return CManager.getContext();
}

void Executor::mainLoop() {
    while (true) {
        std::unique_ptr<ICommand> Command;

        {
            std::unique_lock<std::mutex> Lock(Mtx);

            CV.wait(Lock, [this]() {
                return !CommandQueue.empty() || IsExiting;
            });

            if (CommandQueue.empty() && IsExiting) {
                break;
            }

            Command = std::move(CommandQueue.front());
            CommandQueue.pop_front();
        }

        executeCommand(std::move(Command));
    }
}

void Executor::executeCommand(std::unique_ptr<ICommand>&& Command) {
    if (Command->type() == CommandType::CT_Noop) {
        // No operation.
        executeCommand(static_cast<commands::Noop*>(Command.get()));
    } else if (Command->type() == CommandType::CT_Config) {
        executeCommand(std::unique_ptr<commands::IConfig>(
                    static_cast<commands::IConfig*>(Command.release())));
    } else if (Command->type() == CommandType::CT_GetReady) {
        executeCommand(static_cast<commands::GetReady*>(Command.get()));
    } else if (Command->type() == CommandType::CT_SetPosition) {
        executeCommand(static_cast<commands::SetPosition*>(Command.get()));
    } else if (Command->type() == CommandType::CT_Think) {
        executeCommand(static_cast<commands::Think*>(Command.get()));
    }
}

void Executor::executeCommand(const commands::Noop* Command) {
}

void Executor::executeCommand(std::unique_ptr<commands::IConfig>&& Command) {
    Configs.push(std::move(Command));
}

void Executor::executeCommand(const commands::GetReady* Command) {
    pLogger->printLog("[Executor] command GetReady");
    while (!Configs.empty()) {
        std::unique_ptr ConfigCommand = std::move(Configs.front());
        Configs.pop();

        if (ConfigCommand->configType() == commands::ConfigType::Bool) {
            setConfig(static_cast<const commands::BoolConfig*>(ConfigCommand.get()));
        } else if (ConfigCommand->configType() == commands::ConfigType::Integer) {
            setConfig(static_cast<const commands::IntegerConfig*>(ConfigCommand.get()));
        } else if (ConfigCommand->configType() == commands::ConfigType::Double) {
            setConfig(static_cast<const commands::DoubleConfig*>(ConfigCommand.get()));
        } else if (ConfigCommand->configType() == commands::ConfigType::String) {
            setConfig(static_cast<const commands::StringConfig*>(ConfigCommand.get()));
        }
    }
}

void Executor::executeCommand(const commands::SetPosition* Command) {
    State = std::make_unique<core::State>(io::sfen::StateBuilder::newState(Command->sfen()));
}

void Executor::executeCommand(const commands::Think* Command) {
}

void Executor::setConfig(const commands::BoolConfig* Config) {

}

void Executor::setConfig(const commands::IntegerConfig* Config) {
}

void Executor::setConfig(const commands::DoubleConfig* Config) {
}

void Executor::setConfig(const commands::StringConfig* Config) {
}

} // namespace command
} // namespace engine
} // namespace nshogi
