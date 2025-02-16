//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_SETPOSITION
#define NSHOGI_ENGINE_COMMAND_COMMANDS_SETPOSITION

#include "../command.h"

#include <string>

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class SetPosition : public ICommand {
 public:
    SetPosition(const char* Sfen);

    CommandType type() const;
    const char* sfen() const;

 private:
    std::string _Sfen;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_SETPOSITION
