//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_EVALUATE_BATCH_H
#define NSHOGI_ENGINE_EVALUATE_BATCH_H

#include "evaluator.h"

#include <cstddef>
#include <vector>

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/ml/featurebitboard.h>
#include <nshogi/ml/featurestack.h>
#include <nshogi/ml/featuretype.h>

namespace nshogi {
namespace engine {
namespace evaluate {

template <typename Features>
class Batch {
 public:
    Batch(std::size_t BatchSize, evaluate::Evaluator* Ev);
    Batch(Batch&& B);

    std::size_t add(const core::State& State, const core::StateConfig& Config);
    void doInference();
    void doInferenceNonBlocking();
    void await();
    bool isComputing();

    Evaluator* getEvaluator() {
        return PEvaluator;
    }

    inline const float* getPolicy(std::size_t Index) const {
        return PEvaluator->getPolicy() + 27 * core::NumSquares * Index;
    }

    inline float getWinRate(std::size_t Index) const {
        return PEvaluator->getWinRate()[Index];
    }

    inline float getDrawRate(std::size_t Index) const {
        return PEvaluator->getDrawRate()[Index];
    }

    std::size_t getBatchSizeMax() const;
    std::size_t size() const;

    void reset();

 private:
    const std::size_t BatchSizeMax;
    std::size_t Count;
    std::vector<ml::FeatureBitboard> FeatureStacks;

    Evaluator* PEvaluator;
};

} // namespace evaluate
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_EVALUATE_BATCH_H
