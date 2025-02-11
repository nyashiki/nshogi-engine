//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_THINK
#define NSHOGI_ENGINE_COMMAND_COMMANDS_THINK

#include "../command.h"
#include "../../limit.h"

#include <functional>
#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class Think : public ICommand {
 public:
    Think(Limit Limits[2], std::function<void(core::Move32)> Callback = nullptr);
    ~Think();

    CommandType type() const;
    const Limit* limit() const;
    std::function<void(core::Move32)> callback() const;

 private:
    std::function<void(core::Move32)> CallbackFunc;
    Limit L[2];
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_THINK
