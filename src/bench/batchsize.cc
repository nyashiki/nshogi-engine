//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "batchsize.h"

#if defined(EXECUTOR_TRT)

#include "../infer/trt.h"

#endif

#include "../evaluate/batch.h"
#include "../evaluate/evaluator.h"
#include "../globalconfig.h"

#include <nshogi/core/statebuilder.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/ml/featurestack.h>

#include <chrono>
#include <iostream>

namespace nshogi {
namespace engine {
namespace bench {

void benchBatchSize([[maybe_unused]] const char* WeightPath,
                    [[maybe_unused]] std::size_t Repeat = 1000) {
#if defined(EXECUTOR_TRT)
    std::cout << "Bench batch size for the weight file " << WeightPath
              << " with " << Repeat << " repeats." << std::endl;
    for (uint16_t BatchSize = 60; BatchSize < 160; ++BatchSize) {
        // Setup an evaluator.
        auto TRTInfer =
            infer::TensorRT(0, BatchSize, GlobalConfig::FeatureType::size());
        TRTInfer.load(WeightPath, false);
        evaluate::Evaluator Evaluator(BatchSize, &TRTInfer);
        evaluate::Batch<GlobalConfig::FeatureType> Batch(BatchSize, &Evaluator);

        // Prepare states.
        const auto State = nshogi::core::StateBuilder::getInitialState();
        const core::StateConfig Config;

        for (std::size_t I = 0; I < BatchSize; ++I) {
            Batch.add(State, Config);
        }

        for (std::size_t WarmUp = 0; WarmUp < 4; ++WarmUp) {
            Batch.doInference();
        }

        const auto StartTime = std::chrono::steady_clock::now();

        for (std::size_t I = 0; I < Repeat; ++I) {
            Batch.doInference();
        }

        const auto EndTime = std::chrono::steady_clock::now();
        const auto Duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(EndTime -
                                                                  StartTime)
                .count();

        std::cout << BatchSize << ", " << Duration << ", "
                  << (double)(BatchSize * Repeat) / (double)Duration * 1000.0
                  << std::endl;
    }
#endif
}

} // namespace bench
} // namespace engine
} // namespace nshogi
