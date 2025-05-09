//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_EVALUATE_EVALUATOR_H
#define NSHOGI_ENGINE_EVALUATE_EVALUATOR_H

#include <cstddef>
#include <memory>
#include <vector>

#include "../infer/infer.h"

#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>

namespace nshogi {
namespace engine {
namespace evaluate {

class Evaluator {
 public:
    Evaluator(std::size_t ThreadId, std::size_t FeatureSize,
              std::size_t BatchSize, infer::Infer* In);
    ~Evaluator();

    void computeNonBlocking(std::size_t BatchSize) {
        PInfer->computeNonBlocking(FeatureBitboards, BatchSize, Policy, WinRate,
                                   DrawRate);
    }

    void computeBlocking(std::size_t BatchSize) {
        PInfer->computeBlocking(FeatureBitboards, BatchSize, Policy, WinRate,
                                DrawRate);
    }

    void await() {
        PInfer->await();
    }

    bool isComputing() {
        return PInfer->isComputing();
    }

    inline ml::FeatureBitboard* getFeatureBitboards() {
        return FeatureBitboards;
    }

    inline const float* getPolicy() const {
        return Policy;
    }

    inline const float* getWinRate() const {
        return WinRate;
    }

    inline const float* getDrawRate() const {
        return DrawRate;
    }

    infer::Infer* getInfer() {
        return PInfer;
    }

    void* allocateMemoryByNumaIfAvailable(std::size_t Size) const;
    void freeMemory(void** Memory, std::size_t Size) const;

 private:
    ml::FeatureBitboard* FeatureBitboards;
    float* Policy;
    float* WinRate;
    float* DrawRate;

    infer::Infer* const PInfer;
    const std::size_t MyFeatureSize;
    const std::size_t BatchSizeMax;
    [[maybe_unused]] bool NumaUsed;

    std::vector<int> AvailableNumaNodes;
    [[maybe_unused]] int MyNumaId;
};

} // namespace evaluate
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_EVALUATE_EVALUATOR_H
