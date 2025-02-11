//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "config.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

BoolConfig::BoolConfig(Configurable C_, bool Value)
    : IConfig(C_)
    , V(Value) {
}

ConfigType BoolConfig::configType() const {
    return ConfigType::Bool;
}

bool BoolConfig::value() const {
    return V;
}

IntegerConfig::IntegerConfig(Configurable C_, int64_t Value)
    : IConfig(C_)
    , V(Value) {
}

ConfigType IntegerConfig::configType() const {
    return ConfigType::Integer;
}

int64_t IntegerConfig::value() const {
    return V;
}

DoubleConfig::DoubleConfig(Configurable C_, double Value)
    : IConfig(C_)
    , V(Value) {
}

ConfigType DoubleConfig::configType() const {
    return ConfigType::Double;
}

double DoubleConfig::value() const {
    return V;
}

StringConfig::StringConfig(Configurable C_, const std::string& Value)
    : IConfig(C_)
    , V(Value) {
}

ConfigType StringConfig::configType() const {
    return ConfigType::String;
}

const std::string& StringConfig::value() const {
    return V;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
