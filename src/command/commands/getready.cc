//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "getready.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

GetReady::~GetReady() {
}

CommandType GetReady::type() const {
    return CommandType::CT_GetReady;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
