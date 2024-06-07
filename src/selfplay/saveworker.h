#ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H

#include "../worker/worker.h"

namespace nshogi {
namespace engine {
namespace selfplay {

class SaveWorker : public worker::Worker {
 private:
    bool doTask() override;
};

} // namespace selfpaly
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_SAVEWORKER_H
