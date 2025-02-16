//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "think.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

Think::Think(Limit Limits[2], std::function<void(core::Move32)> Callback) {
    L[0] = Limits[0];
    L[1] = Limits[1];

    CallbackFunc = Callback;
}

Think::~Think() {
}

CommandType Think::type() const {
    return CommandType::CT_Think;
}

const Limit* Think::limit() const {
    return L;
}

std::function<void(core::Move32)> Think::callback() const {
    return CallbackFunc;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
