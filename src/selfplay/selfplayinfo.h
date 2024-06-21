#ifndef NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H
#define NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H

#include <condition_variable>
#include <mutex>
#include <cstdint>

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

 private:
    std::size_t NumOnGoingGames;
    double AverageBatchSize;
    uint64_t InferenceCount;

    mutable std::mutex Mutex;
    mutable std::mutex MutexInference;
    std::condition_variable CV;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SELFPLAYINFO_H
