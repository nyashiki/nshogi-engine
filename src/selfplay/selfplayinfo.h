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

 private:
    std::size_t NumOnGoingGames;
    double AverageBatchSize;
    uint64_t InferenceCount;

    std::atomic<uint64_t> NumCacheHit;
    std::atomic<uint64_t> NumCacheMiss;

    mutable std::mutex Mutex;
    mutable std::mutex MutexInference;
    std::condition_variable CV;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H
