#ifndef NSHOGI_ENGINE_INFER_ZERO_H
#define NSHOGI_ENGINE_INFER_ZERO_H

#include "infer.h"

namespace nshogi {
namespace engine {
namespace infer {

class Zero : public Infer {
 public:
    Zero();
    ~Zero() override;

    void computeNonBlocking(const ml::FeatureBitboard*, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void computeBlocking(const ml::FeatureBitboard*, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void await() override;
    bool isComputing() override;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_ZERO_H
