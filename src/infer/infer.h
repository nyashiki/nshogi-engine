#ifndef NSHOGI_ENGINE_INFER_INFER_H
#define NSHOGI_ENGINE_INFER_INFER_H

#include <nshogi/ml/featurebitboard.h>

namespace nshogi {
namespace engine {
namespace infer {

class Infer {
 public:
    virtual ~Infer() {
    }

    virtual void computeNonBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) = 0;
    virtual void computeBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) = 0;
    virtual void await() = 0;
    virtual bool isComputing() = 0;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_INFER_H
