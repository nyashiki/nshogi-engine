#include "executor.h"

#include "../allocator/allocator.h"
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace command {

Executor::Executor(std::shared_ptr<logger::Logger> Logger)
    : Worker(&Executor::mainLoop, this)
    , IsExiting(false)
    , PLogger(std::move(Logger)) {

    StateConfig = std::make_unique<core::StateConfig>();
    StateConfig->Rule = core::ER_Declare27;
}

Executor::~Executor() {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        IsExiting = true;
    }

    CV.notify_one();
    Worker.join();
}

void Executor::pushCommand(std::shared_ptr<ICommand> Command, bool blocking) {
    {
        if (blocking) {
            Command->blockingMode();
        }
        std::lock_guard<std::mutex> Lock(Mtx);
        CommandQueue.push_back(Command);
    }

    CV.notify_one();
    if (blocking) {
        Command->wait();
    }
}

const Context* Executor::getContext() const {
    return CManager.getContext();
}

void Executor::mainLoop() {
    while (true) {
        std::shared_ptr<ICommand> Command;

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

void Executor::executeCommand(std::shared_ptr<ICommand> Command) {
    if (Command->type() == CommandType::CT_Noop) {
        // No operation.
        executeCommand(static_cast<commands::Noop*>(Command.get()));
    } else if (Command->type() == CommandType::CT_Config) {
        executeCommand(static_cast<commands::IConfig*>(Command.get()));
    } else if (Command->type() == CommandType::CT_GetReady) {
        executeCommand(static_cast<commands::GetReady*>(Command.get()));
    } else if (Command->type() == CommandType::CT_SetPosition) {
        executeCommand(static_cast<commands::SetPosition*>(Command.get()));
    } else if (Command->type() == CommandType::CT_Think) {
        executeCommand(static_cast<commands::Think*>(Command.get()));
    } else if (Command->type() == CommandType::CT_Stop) {
        executeCommand(static_cast<commands::Stop*>(Command.get()));
    }

    Command->setDone();
}

void Executor::executeCommand(const commands::Noop*) {
}

void Executor::executeCommand(const commands::IConfig* Command) {
    if (Command->configType() == commands::ConfigType::Bool) {
        setConfig(static_cast<const commands::BoolConfig*>(Command));
    } else if (Command->configType() == commands::ConfigType::Integer) {
        setConfig(static_cast<const commands::IntegerConfig*>(Command));
    } else if (Command->configType() == commands::ConfigType::Double) {
        setConfig(static_cast<const commands::DoubleConfig*>(Command));
    } else if (Command->configType() == commands::ConfigType::String) {
        setConfig(static_cast<const commands::StringConfig*>(Command));
    }
}

void Executor::executeCommand(const commands::GetReady*) {
    Manager = std::make_unique<mcts::Manager>(CManager.getContext(), PLogger);
    Manager->setIsPonderingEnabled(CManager.getContext()->getPonderingEnabled());
}

void Executor::executeCommand(const commands::SetPosition* Command) {
    State = std::make_unique<core::State>(io::sfen::StateBuilder::newState(Command->sfen()));
}

void Executor::executeCommand(const commands::Think* Command) {
    const Limit* MyLimit = State->getSideToMove() == core::Black ? &Command->limit()[0] : &Command->limit()[1];
    Manager->thinkNextMove(*State, *StateConfig, *MyLimit, Command->callback());
}

void Executor::executeCommand(const commands::Stop*) {
    Manager->interrupt();
}

void Executor::setConfig(const commands::BoolConfig* Config) {
    if (Config->configurable() == commands::Configurable::PonderEnabled) {
        CManager.setPonderingEnabled(Config->value());
    } else if (Config->configurable() == commands::Configurable::BookEnabled) {
        CManager.setBookEnabled(Config->value());
    } else if (Config->configurable() == commands::Configurable::RepetitionBookAllowed) {
        CManager.setRepetitionBookAllowed(Config->value());
    }
}

void Executor::setConfig(const commands::IntegerConfig* Config) {
    if (Config->configurable() == commands::Configurable::MaxPly) {
        StateConfig->MaxPly = (uint16_t)Config->value();
    } else if (Config->configurable() == commands::Configurable::NumGPUs) {
        CManager.setNumGPUs((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::NumSearchThreadsPerGPU) {
        CManager.setNumSearchThreads((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::NumEvaluationThreadsPerGPU) {
        CManager.setNumEvaluationThreadsPerGPU((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::NumCheckmateSearchThreads) {
        CManager.setNumCheckmateSearchThreads((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::BatchSize) {
        CManager.setBatchSize((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::HashMemoryMB) {
        CManager.setAvailableMemoryMB((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::EvalCacheMemoryMB) {
        CManager.setEvalCacheMemoryMB((std::size_t)Config->value());
    } else if (Config->configurable() == commands::Configurable::ThinkingTimeMargin) {
        CManager.setThinkingTimeMargin((uint32_t)Config->value());
    }
}

void Executor::setConfig(const commands::DoubleConfig* Config) {
    if (Config->configurable() == commands::Configurable::BlackDrawValue) {
        StateConfig->BlackDrawValue = (float)Config->value();
    } else if (Config->configurable() == commands::Configurable::WhiteDrawValue) {
        StateConfig->WhiteDrawValue = (float)Config->value();
    }
}

void Executor::setConfig(const commands::StringConfig* Config) {
    if (Config->configurable() == commands::Configurable::WeightPath) {
        CManager.setWeightPath(Config->value());
    } else if (Config->configurable() == commands::Configurable::BookPath) {
        CManager.setBookPath(Config->value());
    }
}

} // namespace command
} // namespace engine
} // namespace nshogi
