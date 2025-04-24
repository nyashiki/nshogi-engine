//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_LOGGER_LOGGER_H
#define NSHOGI_ENGINE_LOGGER_LOGGER_H

#include <cinttypes>
#include <nshogi/core/types.h>
#include <vector>

namespace nshogi {
namespace engine {
namespace logger {

struct PVLog {
    uint32_t ElapsedMilliSeconds = 0;
    core::Color CurrentSideToMove = core::NoColor;
    double WinRate = 0;
    double DrawRate = 0;
    double DrawValue = 0.5;
    int32_t SolvedGameEndPly = 0;
    uint64_t NodesPerSecond = 0;
    uint64_t NumNodes = 0;
    uint64_t HashFullPerMille = 0;
    std::vector<core::Move16> PV;
};

class Logger {
 public:
    virtual ~Logger(){};

    virtual void printPVLog(const PVLog& Log) const = 0;
    virtual void printBestMove(core::Move32 Move) const = 0;
    virtual void printLog(const char* Message) const = 0;

    virtual void setIsInverse(bool Value) = 0;
    void setIsNShogiExtensionLogEnabled(bool Value);

 protected:
    bool IsNShogiExtensionEnabled;
};

} // namespace logger
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_LOGGER_H
