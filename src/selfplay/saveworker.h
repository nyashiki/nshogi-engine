#ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H

#include "framequeue.h"
#include "selfplayinfo.h"
#include "../worker/worker.h"

#include <chrono>
#include <fstream>

namespace nshogi {
namespace engine {
namespace selfplay {

class SaveWorker : public worker::Worker {
 public:
    SaveWorker(SelfplayInfo*, FrameQueue*, FrameQueue*, std::size_t NumSelfplayGames, const char* SavePath);

 private:
    bool doTask() override;
    void updateStatistics(Frame*);
    void printStatistics(bool Force) const;
    void save(Frame*);

    const std::size_t NumSelfplayGamesToStop;

    SelfplayInfo* SInfo;
    FrameQueue* SaveQueue;
    FrameQueue* SearchQueue;
    std::ofstream Ofs;

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
