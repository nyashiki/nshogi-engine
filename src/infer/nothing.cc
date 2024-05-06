#include "nothing.h"

namespace nshogi {
namespace engine {
namespace infer {

Nothing::Nothing() {
}

Nothing::~Nothing() {
}

void Nothing::computeNonBlocking(const ml::FeatureBitboard*, std::size_t, float*, float*, float*) {
}

void Nothing::computeBlocking(const ml::FeatureBitboard*, std::size_t, float*, float*, float*) {
}

void Nothing::await() {
}

bool Nothing::isComputing() {
    return false;
}

} // namespace infer
} // namespace engine
} // namespace nshogi
