#ifndef NSHOGI_ENGINE_EVALUATE_EVALUATOR_H
#define NSHOGI_ENGINE_EVALUATE_EVALUATOR_H


#include <cstddef>
#include <memory>

#include "../infer/infer.h"

#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>


namespace nshogi {
namespace engine {
namespace evaluate {

class Evaluator {
 public:
    Evaluator(std::size_t BatchSize, infer::Infer* In): PInfer(In) {
        Policy = std::make_unique<float[]>(ml::MoveIndexMax * BatchSize);
        WinRate = std::make_unique<float[]>(BatchSize);
        DrawRate = std::make_unique<float[]>(BatchSize);
    }

    void computeNonBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize) {
        PInfer->computeNonBlocking(Features, BatchSize, Policy.get(), WinRate.get(), DrawRate.get());
    }

    void computeBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize) {
        PInfer->computeBlocking(Features, BatchSize, Policy.get(), WinRate.get(), DrawRate.get());
    }

    void await() {
        PInfer->await();
    }

    bool isComputing() {
        return PInfer->isComputing();
    }

    inline const float* getPolicy() const {
        return Policy.get();
    }

    inline const float* getWinRate() const {
        return WinRate.get();
    }

    inline const float* getDrawRate() const {
        return DrawRate.get();
    }

 private:
    std::unique_ptr<float[]> Policy;
    std::unique_ptr<float[]> WinRate;
    std::unique_ptr<float[]> DrawRate;

    infer::Infer* const PInfer;
};

} // namespace evaluate
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_EVALUATE_EVALUATOR_H
