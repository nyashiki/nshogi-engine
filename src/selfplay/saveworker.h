#ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H

#include "framequeue.h"
#include "../worker/worker.h"

#include <chrono>

namespace nshogi {
namespace engine {
namespace selfplay {

class SaveWorker : public worker::Worker {
 public:
    SaveWorker(FrameQueue*, FrameQueue*);

 private:
    bool doTask() override;
    void updateStatistics(Frame*);
    void printStatistics() const;

    FrameQueue* SaveQueue;
    FrameQueue* SearchQueue;

    std::chrono::time_point<std::chrono::steady_clock> StartTime;
    mutable std::chrono::time_point<std::chrono::steady_clock> PreviousPrintTime;
    std::string LatestGame;

    struct {
        uint64_t NumBlackWin = 0;
        uint64_t NumWhiteWin = 0;
        uint64_t NumDraw = 0;
        uint64_t NumDeclare = 0;
        double AveragePly = 0.0;
        double AveragePlyDraw = 0.0;
    } Statistics;
};

} // namespace selfpaly
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
