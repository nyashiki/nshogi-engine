//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "selfplayinfo.h"

#include <cassert>

namespace nshogi {
namespace engine {
namespace selfplay {

SelfplayInfo::SelfplayInfo(std::size_t OnGoingGames)
    : NumOnGoingGames(OnGoingGames)
    , AverageBatchSize(0)
    , InferenceCount(0)
    , NumCacheHit(0)
    , NumCacheMiss(0) {
}

void SelfplayInfo::decrementNumOnGoingGames() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        assert(NumOnGoingGames > 0);
        --NumOnGoingGames;
    }
    CV.notify_all();
}

void SelfplayInfo::waitUntilAllGamesFinished() {
    std::unique_lock<std::mutex> Lock(Mutex);
    CV.wait(Lock, [this]() { return NumOnGoingGames == 0; });
}

std::size_t SelfplayInfo::getNumOnGoinggames() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return NumOnGoingGames;
}

void SelfplayInfo::putBatchSizeStatistics(std::size_t BatchSize) {
    std::lock_guard<std::mutex> Lock(MutexInference);
    AverageBatchSize =
        (AverageBatchSize * (double)InferenceCount + (double)BatchSize) /
        (double)(InferenceCount + 1);
    ++InferenceCount;
}

double SelfplayInfo::getAverageBatchSize() const {
    std::lock_guard<std::mutex> Lock(MutexInference);
    return AverageBatchSize;
}

void SelfplayInfo::incrementCacheHit() {
    NumCacheHit.fetch_add(1, std::memory_order_relaxed);
}

void SelfplayInfo::incrementCacheMiss() {
    NumCacheMiss.fetch_add(1, std::memory_order_relaxed);
}

double SelfplayInfo::getCacheHitRatio() const {
    const uint64_t CH = NumCacheHit.load(std::memory_order_relaxed);
    const uint64_t CM = NumCacheMiss.load(std::memory_order_relaxed);
    const uint64_t N = CH + CM;

    return (N == 0) ? 0.0 : ((double)CH / (double)N);
}

uint64_t SelfplayInfo::numBlackWin() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return NumBlackWin;
}

uint64_t SelfplayInfo::numWhiteWin() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return NumWhiteWin;
}

uint64_t SelfplayInfo::numDraw() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return NumDraw;
}

uint64_t SelfplayInfo::numGenerated() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return NumBlackWin + NumWhiteWin + NumDraw;
}

uint64_t SelfplayInfo::numDeclare() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return NumDeclare;
}

double SelfplayInfo::averagePly() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return AveragePly;
}

double SelfplayInfo::averagePlyDraw() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    return AveragePlyDraw;
}

double SelfplayInfo::blackWinRate() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    if (NumBlackWin + NumWhiteWin + NumDraw == 0) {
        return 0.0;
    }
    return (double)NumBlackWin / (double)(NumBlackWin + NumWhiteWin + NumDraw);
}

double SelfplayInfo::whiteWinRate() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    if (NumBlackWin + NumWhiteWin + NumDraw == 0) {
        return 0.0;
    }
    return (double)NumWhiteWin / (double)(NumBlackWin + NumWhiteWin + NumDraw);
}

double SelfplayInfo::drawRate() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    if (NumBlackWin + NumWhiteWin + NumDraw == 0) {
        return 0.0;
    }
    return (double)NumDraw / (double)(NumBlackWin + NumWhiteWin + NumDraw);
}

double SelfplayInfo::declareRate() const {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    if (NumBlackWin + NumWhiteWin + NumDraw == 0) {
        return 0.0;
    }
    return (double)NumDeclare / (double)(NumBlackWin + NumWhiteWin + NumDraw);
}

void SelfplayInfo::incrementBlackWin() {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    ++NumBlackWin;
}

void SelfplayInfo::incrementWhiteWin() {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    ++NumWhiteWin;
}

void SelfplayInfo::incrementDraw() {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    ++NumDraw;
}

void SelfplayInfo::incrementDeclare() {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    ++NumDeclare;
}

void SelfplayInfo::updateAveragePly(uint16_t Ply) {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    const uint64_t N = NumBlackWin + NumWhiteWin;
    AveragePly = (AveragePly * (double)N + (double)Ply) / (double)(N + 1);
}

void SelfplayInfo::updateAveragePlyDraw(uint16_t Ply) {
    std::lock_guard<std::mutex> Lock(MutexStatistics);
    AveragePlyDraw = (AveragePlyDraw * (double)NumDraw + (double)Ply) / (double)(NumDraw + 1);
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
