//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "batch.h"
#include "evaluator.h"
#include "preset.h"

#include <cstddef>
#include <iostream>
#include <nshogi/ml/featurestack.h>

namespace nshogi {
namespace engine {
namespace evaluate {

template <typename Features>
Batch<Features>::Batch(std::size_t BatchSize, evaluate::Evaluator* Ev)
    : BatchSizeMax(BatchSize)
    , Count(0)
    , PEvaluator(Ev) {

    FeatureStacks.reserve(BatchSizeMax * Features::size());
}

template <typename Features>
Batch<Features>::Batch(Batch&& B)
    : BatchSizeMax(B.BatchSizeMax)
    , Count(B.Count)
    , FeatureStacks(std::move(B.FeatureStacks))
    , PEvaluator(B.PEvaluator) {

    B.PEvaluator = nullptr;
}

template <typename Features>
void Batch<Features>::doInference() {
    assert(Count > 0);
    assert(Count <= BatchSizeMax);

    PEvaluator->computeBlocking(Count);
}

template <typename Features>
void Batch<Features>::doInferenceNonBlocking() {
    assert(Count > 0);
    assert(Count <= BatchSizeMax);

    PEvaluator->computeNonBlocking(Count);
}

template <typename Features>
void Batch<Features>::await() {
    PEvaluator->await();
}

template <typename Features>
bool Batch<Features>::isComputing() {
    return PEvaluator->isComputing();
}

template <typename Features>
std::size_t Batch<Features>::size() const {
    return Count;
}

template <typename Features>
std::size_t Batch<Features>::getBatchSizeMax() const {
    return BatchSizeMax;
}

template <typename Features>
std::size_t Batch<Features>::add(const core::State& State,
                                 const core::StateConfig& Config) {
    Features::constructAt(FeatureStacks.data() + Count * Features::size(),
                          State, Config);

    return Count++;
}

template <typename Features>
void Batch<Features>::reset() {
    FeatureStacks.clear();

    Count = 0;
}

template class Batch<preset::SimpleFeatures>;
template class Batch<preset::CustomFeaturesV1>;

} // namespace evaluate
} // namespace engine
} // namespace nshogi
