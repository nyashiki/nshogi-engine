//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H
#define NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H

#include "../worker/worker.h"
#include "checkmatequeue.h"

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

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H
