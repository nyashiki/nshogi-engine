#include "random.h"

#include <cstddef>
#include <cstdint>
#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>

namespace nshogi {
namespace engine {
namespace infer {

Random::Random(uint64_t Seed): Rng(Seed) {
}

Random::~Random() {
}

void Random::computeNonBlocking([[maybe_unused]] const ml::FeatureBitboard* Features,
        std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) {

    static std::uniform_real_distribution<float> Distribution(0, 1);

    for (std::size_t I = 0; I < BatchSize; ++I) {
        for (std::size_t J = 0; J < ml::MoveIndexMax; ++J) {
            DstPolicy[I * ml::MoveIndexMax + J] = Distribution(Rng);
        }

        DstWinRate[I] = Distribution(Rng);
        DstDrawRate[I] = Distribution(Rng);
    }
}

void Random::computeBlocking(const ml::FeatureBitboard* Features,
        std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) {
    computeNonBlocking(Features, BatchSize, DstPolicy, DstWinRate, DstDrawRate);
    await();
}

void Random::await() {
    // Dummy.
}

bool Random::isComputing() {
    return false;
}

} // namespace infer
} // namespace engine
} // namespace nshogi
