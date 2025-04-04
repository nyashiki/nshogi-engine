//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_PROTOCOL_USILOGGER_H
#define NSHOGI_ENGINE_PROTOCOL_USILOGGER_H

#include "../logger/logger.h"
#include <cstdint>
#include <iostream>
#include <mutex>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

class USILogger : public logger::Logger {
 public:
    USILogger();
    ~USILogger() = default;

    void printPVLog(const logger::PVLog& Log) const override;
    void printBestMove(core::Move32 Move) const override;

    void printLog(const char* Message) const override;

    template <typename... Ts>
    void printLog(Ts... Args) const {
        printRawMessage("info string ", Args...);
    }

    template <typename... Ts>
    void printRawMessage(const Ts&... Args) const {
        std::lock_guard<std::mutex> Lock(Mtx);
        (std::cout << ... << Args) << std::endl;
    }

    void setIsInverse(bool Value) override;

 private:
    int32_t getScoreFromWinRate(double WinRate, double DrawRate,
                                double DrawValue) const;

    bool IsInverse;
    mutable std::mutex Mtx;
};

} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_PROTOCOL_USILOGGER_H
