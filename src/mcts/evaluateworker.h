#ifndef NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H
#define NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H

#include "node.h"
#include "evalcache.h"
#include "evaluatequeue.h"
#include "../evaluate/evaluator.h"
#include "../infer/infer.h"
#include "../worker/worker.h"
#include "../context.h"

#include <nshogi/ml/featurebitboard.h>
#include <nshogi/ml/common.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
class EvaluateWorker : public worker::Worker {
 public:
    EvaluateWorker(const Context*, std::size_t GPUId, std::size_t BatchSize, EvaluationQueue<Features>*, EvalCache*);
    ~EvaluateWorker();

 private:
    static constexpr std::size_t SEQUENTIAL_SKIP_THRESHOLD = 3;

    void initializationTask() override;
    bool doTask() override;
    void getBatch();
    void flattenFeatures(std::size_t BatchSize);
    void doInference(std::size_t BatchSize);
    void feedResults(std::size_t BatchSize);
    void feedResult(core::Color, Node*, const float* Policy, float WinRate, float DrawRate, uint64_t Hash);

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

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H
