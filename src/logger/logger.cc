//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "logger.h"

namespace nshogi {
namespace engine {
namespace logger {

void Logger::setIsNShogiExtensionLogEnabled(bool Value) {
    IsNShogiExtensionEnabled = Value;
}

} // namespace logger
} // namespace engine
} // namespace nshogi
