#include "usilogger.h"
#include "../globalconfig.h"

#include <cstdint>
#include <cmath>
#include <mutex>
#include <iostream>

#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

USILogger::USILogger()
    : SFType(ScoreFormatType::CentiPawn) {
    IsInverse = false;
}

void USILogger::printPVLog(const logger::PVLog& Log) const {
    std::lock_guard<std::mutex> Lock(Mtx);

    std::cout << "info depth " << Log.PV.size() <<  " time " << Log.ElapsedMilliSeconds;

    if (Log.SolvedGameEndPly != 0) {
        std::cout << " score mate " << Log.SolvedGameEndPly;
    } else {
        if (SFType == ScoreFormatType::WinDraw) {
            const double WinRate = (1.0 - Log.DrawRate) * ((IsInverse)? (1.0 - Log.WinRate) : Log.WinRate);
            std::cout << " score windraw " << WinRate << " " << Log.DrawRate;
        } else {
            int32_t Score = getScoreFromWinRate(Log.WinRate, Log.DrawRate, Log.DrawValue);
            if (IsInverse) {
                Score = -Score;
            }
            std::cout << " score cp " << Score;
        }
    }

    std::cout << " nodes " << Log.NumNodes << " nps " << Log.NodesPerSecond
            << " hashfull " << Log.HashFullPerMille << " pv";

    for (const auto& Move : Log.PV) {
        std::cout << " " << io::sfen::move16ToSfen(Move);
    }

    std::cout << std::endl;
}

void USILogger::printBestMove(core::Move32 Move) const {
    std::lock_guard<std::mutex> Lock(Mtx);
    std::cout << "bestmove " << nshogi::io::sfen::move32ToSfen(Move) << std::endl;
}

void USILogger::printLog(const char* Message) const {
    std::lock_guard<std::mutex> Lock(Mtx);
    std::cout << "info string " << Message << std::endl;
}

void USILogger::setIsInverse(bool Value) {
    IsInverse = Value;
}

void USILogger::setScoreFormatType(ScoreFormatType Type) {
    SFType = Type;
}

int32_t USILogger::getScoreFromWinRate(double WinRate, double DrawRate, double DrawValue) const {
    const double PonanzaConstant = 600;

    const double WinRateConsideringDraw = DrawRate * DrawValue + (1 - DrawRate) * WinRate;
    if (WinRateConsideringDraw == 0) {
        return -9999;
    }

    return (int32_t)(-PonanzaConstant * std::log(1.0 / WinRateConsideringDraw - 1.0));
}


} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi
