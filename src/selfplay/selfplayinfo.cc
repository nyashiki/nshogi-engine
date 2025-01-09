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
        (AverageBatchSize * (double)InferenceCount + (double)BatchSize) / (double)(InferenceCount + 1);
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

} // namespace selfplay
} // namespace engine
} // namespace nshogi
