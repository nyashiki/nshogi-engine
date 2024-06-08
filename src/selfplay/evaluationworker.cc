#include "evaluationworker.h"
#include "../evaluate/preset.h"

namespace nshogi {
namespace engine {
namespace selfplay {

EvaluationWorker::EvaluationWorker(std::size_t BSize, FrameQueue* EQ, FrameQueue* SQ)
    : worker::Worker(true)
    , BatchSize(BSize)
    , EvaluationQueue(EQ)
    , SearchQueue(SQ) {
    spawnThread();
}

bool EvaluationWorker::doTask() {
    auto Tasks = EvaluationQueue->get(BatchSize);

    if (BatchSize == 0) {
        std::this_thread::yield();
        return false;
    }

    for (std::size_t I = 0; I < BatchSize; ++I) {
        evaluate::preset::CustomFeaturesV1::constructAt(
            FeatureBitboards + I * evaluate::preset::CustomFeaturesV1::size(),
            *Tasks.at(I)->getState(),
            *Tasks.at(I)->getStateConfig());
    }

    Evaluator->computeBlocking(FeatureBitboards, BatchSize);

    for (std::size_t I = 0; I < BatchSize; ++I) {
        auto&& F = std::move(Tasks.at(I));
        F->setEvaluation(
                Evaluator->getPolicy() + 27 * core::NumSquares * I,
                Evaluator->getWinRate()[I],
                Evaluator->getDrawRate()[I]);

        F->setPhase(SelfplayPhase::Backpropagation);
        SearchQueue->add(std::move(F));
    }

    return false;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
