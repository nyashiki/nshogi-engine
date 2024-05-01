#ifndef NSHOGI_ENGINE_PROTOCOL_USILOGGER_H
#define NSHOGI_ENGINE_PROTOCOL_USILOGGER_H

#include "../logger/logger.h"
#include <iostream>
#include <mutex>
#include <cstdint>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {


class USILogger : public logger::Logger {
 public:
    enum class ScoreFormatType {
        CentiPawn,
        WinDraw,
    };

    USILogger();

    void printPVLog(const logger::PVLog& Log) const override;
    void printBestMove(const core::Move32& Move) const override;

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

    void setIsInverse(bool Value);
    void setScoreFormatType(ScoreFormatType);

 public:
    int32_t getScoreFromWinRate(double WinRate, double DrawRate, double DrawValue) const;
    bool IsInverse;
    ScoreFormatType SFType;

    mutable std::mutex Mtx;
};


} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_PROTOCOL_USILOGGER_H
