#ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H

#include "framequeue.h"
#include "../worker/worker.h"

namespace nshogi {
namespace engine {
namespace selfplay {

class SaveWorker : public worker::Worker {
 public:
    SaveWorker(FrameQueue*, FrameQueue*);

 private:
    bool doTask() override;

    FrameQueue* SaveQueue;
    FrameQueue* SearchQueue;
};

} // namespace selfpaly
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
