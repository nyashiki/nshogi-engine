#ifndef NSHOGI_ENGINE_INFER_NOTHING_H
#define NSHOGI_ENGINE_INFER_NOTHING_H

#include "infer.h"

namespace nshogi {
namespace engine {
namespace infer {

class Nothing : public Infer {
 public:
    Nothing();
    ~Nothing() override;

    void computeNonBlocking(const ml::FeatureBitboard*, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void computeBlocking(const ml::FeatureBitboard*, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void await() override;
    bool isComputing() override;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_NOTHING_H
