//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_EVALUATIONWORKER_H
#define NSHOGI_ENGINE_MCTS_EVALUATIONWORKER_H

#include "../context.h"
#include "../evaluate/evaluator.h"
#include "../infer/infer.h"
#include "../worker/worker.h"
#include "evalcache.h"
#include "evaluationqueue.h"
#include "node.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>
#include <thread>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class EvaluationWorker : public worker::Worker {
 public:
    EvaluationWorker(const Context*, std::size_t GPUId, std::size_t BatchSize,
                   EvaluationQueue<Features>*, EvalCache*);
    ~EvaluationWorker();

 private:
    static constexpr std::size_t SEQUENTIAL_SKIP_THRESHOLD = 3;

    void initializationTask() override;
    bool doTask() override;
    void getBatch();
    void flattenFeatures(std::size_t BatchSize);
    void doInference(std::size_t BatchSize);
    void feedResults(std::size_t BatchSize);
    void feedResult(core::Color, Node*, const float* Policy, float WinRate,
                    float DrawRate, uint64_t Hash);

    const Context* PContext;

    const std::size_t BatchSizeMax;
    EvaluationQueue<Features>* const EQueue;
    EvalCache* const ECache;

    std::unique_ptr<infer::Infer> Infer;
    std::unique_ptr<evaluate::Evaluator> Evaluator;
    std::size_t GPUId_;

    ml::FeatureBitboard* FeatureBitboards;
    float LegalPolicy[ml::MoveIndexMax];

    std::vector<core::Color> PendingSideToMoves;
    std::vector<Node*> PendingNodes;
    std::vector<Features> PendingFeatures;
    std::vector<uint64_t> PendingHashes;
    std::size_t SequentialSkip;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALUATIONWORKER_H
