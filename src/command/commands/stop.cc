//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "stop.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

CommandType Stop::type() const {
    return CommandType::CT_Stop;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
