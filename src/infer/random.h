//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_INFER_RANDOM_H
#define NSHOGI_ENGINE_INFER_RANDOM_H

#include <cstddef>
#include <cstdint>
#include <random>

#include "infer.h"

namespace nshogi {
namespace engine {
namespace infer {

class Random : public Infer {
 public:
    Random(uint64_t Seed);

    ~Random() override;

    void computeNonBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void computeBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void await() override;
    bool isComputing() override;

 private:
    std::mt19937_64 Rng;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_RANDOM_H
