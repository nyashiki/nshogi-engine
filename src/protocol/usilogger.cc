//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "usilogger.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>

#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

USILogger::USILogger() {
    IsInverse = false;
}

void USILogger::printPVLog(const logger::PVLog& Log) const {
    std::lock_guard<std::mutex> Lock(Mtx);

    std::cout << "info depth " << Log.PV.size() << " time "
              << Log.ElapsedMilliSeconds;

    if (Log.SolvedGameEndPly != 0) {
        std::cout << " score mate " << Log.SolvedGameEndPly;
    } else {
        int32_t Score =
            getScoreFromWinRate(Log.WinRate, Log.DrawRate, Log.DrawValue);
        if (IsInverse) {
            Score = -Score;
        }
        std::cout << " score cp " << Score;
    }

    std::cout << " nodes " << Log.NumNodes << " nps " << Log.NodesPerSecond
              << " hashfull " << Log.HashFullPerMille << " pv";

    for (const auto& Move : Log.PV) {
        std::cout << " " << io::sfen::move16ToSfen(Move);
    }

    std::cout << std::endl;

    if (IsNShogiExtensionEnabled) {
        const double WinRate = (1.0 - Log.DrawRate) * Log.WinRate;
        const double BlackWinRate = (Log.CurrentSideToMove == core::Black)
                                        ? WinRate
                                        : (1.0 - Log.DrawRate - WinRate);
        const double WhiteWinRate = 1.0 - Log.DrawRate - BlackWinRate;
        std::cout << "info nshogiext black_win_rate " << BlackWinRate
                  << " draw_rate " << Log.DrawRate << " white_win_rate "
                  << WhiteWinRate << std::endl;
    }
}

void USILogger::printBestMove(core::Move32 Move) const {
    std::lock_guard<std::mutex> Lock(Mtx);
    std::cout << "bestmove " << nshogi::io::sfen::move32ToSfen(Move)
              << std::endl;
}

void USILogger::printLog(const char* Message) const {
    std::lock_guard<std::mutex> Lock(Mtx);
    std::cout << "info string " << Message << std::endl;
}

void USILogger::setIsInverse(bool Value) {
    IsInverse = Value;
}

int32_t USILogger::getScoreFromWinRate(double WinRate, double DrawRate,
                                       double DrawValue) const {
    const double PonanzaConstant = 600;

    const double WinRateConsideringDraw =
        DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
    if (WinRateConsideringDraw == 0.0) {
        return -9999;
    }

    return (int32_t)(-PonanzaConstant *
                     std::log(1.0 / WinRateConsideringDraw - 1.0));
}

} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi
