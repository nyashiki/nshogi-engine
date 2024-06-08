#include "saveworker.h"

namespace nshogi {
namespace engine {
namespace selfplay {

SaveWorker::SaveWorker(FrameQueue* SVQ, FrameQueue* SCQ)
    : worker::Worker(true)
    , SaveQueue(SVQ)
    , SearchQueue(SCQ) {

    spawnThread();
}

bool SaveWorker::doTask() {
    auto Tasks = SaveQueue->getAll();

    while (!Tasks.empty()) {
        auto Task = std::move(Tasks.front());
        Tasks.pop();

        assert(Task->getPhase() == SelfplayPhase::Save);
        Task->setPhase(SelfplayPhase::Initialization);
        SearchQueue->add(std::move(Task));
    }

    return false;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
