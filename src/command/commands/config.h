//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_CONFIG
#define NSHOGI_ENGINE_COMMAND_COMMANDS_CONFIG

#include "../command.h"

#include <cstdint>
#include <string>

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

enum class ConfigType {
    Bool,
    Integer,
    Double,
    String,
};

enum class Configurable {
    // Bool config.
    PonderEnabled,
    BookEnabled,
    RepetitionBookAllowed,

    // Integer config.
    MaxPly,
    NumGPUs,
    NumSearchThreadsPerGPU,
    NumEvaluationThreadsPerGPU,
    NumCheckmateSearchThreads,
    BatchSize,
    HashMemoryMB,
    EvalCacheMemoryMB,
    ThinkingTimeMargin,
    MinimumThinkingTime,
    MaximumThinkingTime,

    // Double config.
    BlackDrawValue,
    WhiteDrawValue,

    // String config.
    WeightPath,
    BookPath,
};

class IConfig : public ICommand {
 public:
    virtual ~IConfig() {
    }

    CommandType type() const {
        return CommandType::CT_Config;
    }

    Configurable configurable() const {
        return C;
    }

    virtual ConfigType configType() const = 0;

 protected:
    IConfig(Configurable Conf)
        : C(Conf) {
    }

    Configurable C;
};

class BoolConfig : public IConfig {
 public:
    BoolConfig(Configurable, bool Value);
    ConfigType configType() const;
    bool value() const;

 private:
    bool V;
};

class IntegerConfig : public IConfig {
 public:
    IntegerConfig(Configurable, int64_t Value);
    ConfigType configType() const;
    int64_t value() const;

 private:
    int64_t V;
};

class DoubleConfig : public IConfig {
 public:
    DoubleConfig(Configurable, double Value);
    ConfigType configType() const;
    double value() const;

 private:
    double V;
};

class StringConfig : public IConfig {
 public:
    StringConfig(Configurable, const std::string& Value);
    ConfigType configType() const;
    const std::string& value() const;

 private:
    std::string V;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_CONFIG
