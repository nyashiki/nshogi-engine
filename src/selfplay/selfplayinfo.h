//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H
#define NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace nshogi {
namespace engine {
namespace selfplay {

class SelfplayInfo {
 public:
    SelfplayInfo(std::size_t OnGoingGames);

    void decrementNumOnGoingGames();
    void waitUntilAllGamesFinished();
    std::size_t getNumOnGoinggames() const;

    void putBatchSizeStatistics(std::size_t BatchSize);
    double getAverageBatchSize() const;

    void incrementCacheHit();
    void incrementCacheMiss();
    double getCacheHitRatio() const;

    uint64_t numBlackWin() const;
    uint64_t numWhiteWin() const;
    uint64_t numDraw() const;
    uint64_t numGenerated() const;
    uint64_t numDeclare() const;
    double averagePly() const;
    double averagePlyDraw() const;

    double blackWinRate() const;
    double whiteWinRate() const;
    double drawRate() const;
    double declareRate() const;

    void incrementBlackWin();
    void incrementWhiteWin();
    void incrementDraw();
    void incrementDeclare();
    void updateAveragePly(uint16_t Ply);
    void updateAveragePlyDraw(uint16_t Ply);

 private:
    std::size_t NumOnGoingGames;
    double AverageBatchSize;
    uint64_t InferenceCount;

    std::atomic<uint64_t> NumCacheHit;
    std::atomic<uint64_t> NumCacheMiss;

    // Game statistics.
    uint64_t NumBlackWin;
    uint64_t NumWhiteWin;
    uint64_t NumDraw;
    uint64_t NumDeclare;
    double AveragePly;
    double AveragePlyDraw;

    mutable std::mutex Mutex;
    mutable std::mutex MutexInference;
    mutable std::mutex MutexStatistics;
    std::condition_variable CV;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H
