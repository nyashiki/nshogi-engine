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

#include "../infer/infer.h"

#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>

#endif

namespace nshogi {
namespace engine {
namespace evaluate {

class Evaluator {
 public:
    Evaluator(std::size_t BatchSize, infer::Infer* In)
        : PInfer(In) {
#ifdef CUDA_ENABLED
        cudaMallocHost(&Policy, ml::MoveIndexMax * BatchSize * sizeof(float));
        cudaMallocHost(&WinRate, BatchSize * sizeof(float));
        cudaMallocHost(&DrawRate, BatchSize * sizeof(float));
#else
        Policy = new float[ml::MoveIndexMax * BatchSize];
        WinRate = new float[BatchSize];
        DrawRate = new float[BatchSize];
#endif
    }

    ~Evaluator() {
#ifdef CUDA_ENABLED
        cudaFree(Policy);
        cudaFree(WinRate);
        cudaFree(DrawRate);
#else
        delete[] Policy;
        delete[] WinRate;
        delete[] DrawRate;
#endif
    }

    void computeNonBlocking(const ml::FeatureBitboard* Features,
                            std::size_t BatchSize) {
        PInfer->computeNonBlocking(Features, BatchSize, Policy, WinRate,
                                   DrawRate);
    }

    void computeBlocking(const ml::FeatureBitboard* Features,
                         std::size_t BatchSize) {
        PInfer->computeBlocking(Features, BatchSize, Policy, WinRate, DrawRate);
    }

    void await() {
        PInfer->await();
    }

    bool isComputing() {
        return PInfer->isComputing();
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

 private:
    float* Policy;
    float* WinRate;
    float* DrawRate;

    infer::Infer* const PInfer;
};

} // namespace evaluate
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_EVALUATE_EVALUATOR_H
