#ifndef NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H
#define NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H

#include "node.h"
#include "evaluatequeue.h"
#include "../evaluate/evaluator.h"
#include "../worker/worker.h"

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
    EvaluateWorker(std::size_t BatchSize, EvaluationQueue<Features>*, evaluate::Evaluator*);
    ~EvaluateWorker();

    // void start();
    // void stop();

 private:
    static constexpr std::size_t SEQUENTIAL_SKIP_THRESHOLD = 3;

    // void mainLoop();
    bool doTask() override;
    void getBatch();
    void flattenFeatures(const std::vector<Features>&);
    void doInference(std::size_t BatchSize);
    void feedResults(const std::vector<core::Color>&, const std::vector<Node*>&);
    void feedResult(core::Color, Node*, const float* Policy, float WinRate, float DrawRate);

    const std::size_t BatchSizeMax;
    EvaluationQueue<Features>* const EQueue;
    evaluate::Evaluator* const Evaluator;

    std::vector<ml::FeatureBitboard> FeatureBitboards;
    float LegalPolicy[ml::MoveIndexMax];

    std::vector<core::Color> PendingSideToMoves;
    std::vector<Node*> PendingNodes;
    std::vector<Features> PendingFeatures;
    std::size_t SequentialSkip;

    // std::mutex Mutex;
    // std::condition_variable CV;
    // std::atomic<bool> IsRunnning;
    // std::atomic<bool> IsThreadWorking;
    // std::atomic<bool> IsExiting;

    // std::thread Worker;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALUATEWORKER_H
