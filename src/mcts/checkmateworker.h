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

#include "statistics.h"
#include "checkmatequeue.h"
#include "../worker/worker.h"
#include <nshogi/solver/dfpn.h>

namespace nshogi {
namespace engine {
namespace mcts {

class CheckmateWorker : public worker::Worker {
 public:
    CheckmateWorker(std::size_t Id, CheckmateQueue*, Statistics*);
    ~CheckmateWorker();

 private:
    const std::size_t MyId;
    bool doTask() override;

    solver::dfpn::Solver DfPnSolver;
    CheckmateQueue* PCheckmateQueue;
    Statistics* PStat;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_CHECKMATEWORKER_H
