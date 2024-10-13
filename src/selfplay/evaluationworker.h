#ifndef NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H
#define NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H

#include "framequeue.h"
#include "selfplayinfo.h"
#include "../worker/worker.h"
#include "../infer/infer.h"
#include "../evaluate/evaluator.h"

namespace nshogi {
namespace engine {
namespace selfplay {

class EvaluationWorker : public worker::Worker {
 public:
    EvaluationWorker(std::size_t GPUId, std::size_t, const char* WeightPath, FrameQueue*, FrameQueue*, SelfplayInfo*);
    ~EvaluationWorker();

 private:
    void initializationTask() override;
    bool doTask() override;

    void prepareInfer(std::size_t, const char* WeightPath);
    void allocate();

    std::unique_ptr<infer::Infer> Infer;
    std::unique_ptr<evaluate::Evaluator> Evaluator;
    const std::size_t BatchSize;
    FrameQueue* EvaluationQueue;
    FrameQueue* SearchQueue;
    SelfplayInfo* SInfo;

    ml::FeatureBitboard* FeatureBitboards;
    std::vector<std::unique_ptr<Frame>> Tasks;
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_EVALUATIONWORKER_H
