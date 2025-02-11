//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_COMMAND_COMMAND_H
#define NSHOGI_ENGINE_COMMAND_COMMAND_H

#include <condition_variable>
#include <memory>
#include <mutex>

#include <cassert>

namespace nshogi {
namespace engine {
namespace command {

enum class CommandType {
    CT_Noop,
    CT_Config,
    CT_GetReady,
    CT_SetPosition,
    CT_Think,
    CT_Stop,
};

class ICommand {
 public:
    ICommand()
        : IsDone(false)
        , BlockingMode(false) {
    }

    virtual ~ICommand() {
    }

    virtual CommandType type() const = 0;

    bool isDone() const {
        return IsDone;
    }

    void setDone() {
        if (BlockingMode) {
            std::lock_guard<std::mutex> Lock(Mtx);
            IsDone = true;
            CV.notify_all();
        } else {
            IsDone = true;
        }
    }

    void blockingMode() {
        BlockingMode = true;
    }

    void wait() {
        assert(BlockingMode);

        std::unique_lock<std::mutex> Lock(Mtx);
        CV.wait(Lock, [this]() { return IsDone; });
    }

 private:
    std::mutex Mtx;
    std::condition_variable CV;
    bool IsDone;
    bool BlockingMode;
};

} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMAND_H
