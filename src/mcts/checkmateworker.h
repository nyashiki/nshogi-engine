#ifndef NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H
#define NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H

#include "checkmatequeue.h"
#include "../worker/worker.h"

namespace nshogi {
namespace engine {
namespace mcts {

class CheckmateWorker : public worker::Worker {
 public:
    CheckmateWorker(CheckmateQueue*);
    ~CheckmateWorker();

 private:
    bool doTask() override;

    const int SolverDepth;
    CheckmateQueue* PCheckmateQueue;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif  // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H
