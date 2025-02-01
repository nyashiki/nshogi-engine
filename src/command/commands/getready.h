//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_GETREADY
#define NSHOGI_ENGINE_COMMAND_COMMANDS_GETREADY

#include "../command.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class GetReady : public ICommand {
 public:
    ~GetReady();

    CommandType type() const;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_GETREADY
