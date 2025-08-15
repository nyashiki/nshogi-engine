//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_FEEDWORKER_H_
#define NSHOGI_ENGINE_MCTS_FEEDWORKER_H_

#include "feedqueue.h"
#include "evalcache.h"
#include "../worker/worker.h"
#include "../context.h"

#include <memory>

#include <nshogi/ml/common.h>

namespace nshogi {
namespace engine {
namespace mcts {

class FeedWorker : public worker::Worker {
 public:
    FeedWorker(const Context*, FeedQueue*, EvalCache*);

    bool doTask() override;

 private:
    void feedResults(std::unique_ptr<Batch>&&);

    template <bool NaNFallbackEnabled>
    void feedResult(
        core::Color SideToMove,
        Node* N,
        const float* Policy,
        float WinRate,
        float DrawRate,
        uint64_t Hash
    );

    const Context* PContext;
    FeedQueue* Queue;
    EvalCache* ECache;
    float LegalPolicy[ml::MoveIndexMax];
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_FEEDWORKER_H_
