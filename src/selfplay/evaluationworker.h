//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H

#include "../evaluate/evaluator.h"
#include "../infer/infer.h"
#include "../worker/worker.h"
#include "framequeue.h"
#include "selfplayinfo.h"

namespace nshogi {
namespace engine {
namespace selfplay {

class EvaluationWorker : public worker::Worker {
 public:
    EvaluationWorker(std::size_t ThreadId, std::size_t GPUId, std::size_t,
                     const char* WeightPath, FrameQueue*, FrameQueue*,
                     SelfplayInfo*);
    ~EvaluationWorker();

 private:
    void initializationTask() override;
    bool doTask() override;

    void prepareInfer(std::size_t ThreadId, std::size_t GPUId,
                      const char* WeightPath);

    std::unique_ptr<infer::Infer> Infer;
    std::unique_ptr<evaluate::Evaluator> Evaluator;
    const std::size_t BatchSize;
    FrameQueue* EvaluationQueue;
    FrameQueue* SearchQueue;
    SelfplayInfo* SInfo;

    std::vector<std::unique_ptr<Frame>> Tasks;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H
