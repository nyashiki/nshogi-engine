//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evaluationworker.h"
#include "../evaluate/preset.h"

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>

#endif

#ifdef EXECUTOR_ZERO

#include "../infer/zero.h"

#endif

#ifdef EXECUTOR_NOTHING

#include "../infer/nothing.h"

#endif

#ifdef EXECUTOR_RANDOM

#include "../infer/random.h"

#endif

#ifdef EXECUTOR_TRT

#include "../infer/trt.h"

#endif

namespace nshogi {
namespace engine {
namespace selfplay {

EvaluationWorker::EvaluationWorker([[ maybe_unused ]] std::size_t GPUId, std::size_t BSize, [[ maybe_unused ]] const char* WeightPath, FrameQueue* EQ, FrameQueue* SQ, SelfplayInfo* SI)
    : worker::Worker(true)
    , BatchSize(BSize)
    , EvaluationQueue(EQ)
    , SearchQueue(SQ)
    , SInfo(SI) {

    prepareInfer(GPUId, WeightPath);
    allocate();

    spawnThread();
}

EvaluationWorker::~EvaluationWorker() {
#ifdef CUDA_ENABLED
    cudaFree(FeatureBitboards);
#else
    delete[] FeatureBitboards;
#endif
}

void EvaluationWorker::initializationTask() {
#if defined(EXECUTOR_TRT)
    auto TRTInfer = reinterpret_cast<infer::TensorRT*>(Infer.get());
    TRTInfer->resetGPU();
#endif
}

bool EvaluationWorker::doTask() {
    const std::size_t MAX_TRIAL = 4 * BatchSize;

    for (std::size_t Counter = 0; Counter < MAX_TRIAL; ++Counter) {
        Tasks = EvaluationQueue->get(BatchSize, false, Counter == MAX_TRIAL - 1);

        if (Tasks.size() > 0) {
            break;
        }

        std::this_thread::yield();
    }

    if (Tasks.size() == 0) {
        return false;
    }

    for (std::size_t I = 0; I < Tasks.size(); ++I) {
        evaluate::preset::CustomFeaturesV1::constructAt(
            FeatureBitboards + I * evaluate::preset::CustomFeaturesV1::size(),
            *Tasks.at(I)->getState(),
            *Tasks.at(I)->getStateConfig());
    }

    Evaluator->computeBlocking(FeatureBitboards, Tasks.size());

    for (std::size_t I = 0; I < Tasks.size(); ++I) {
        Tasks.at(I)->setEvaluation<false>(
                Evaluator->getPolicy() + 27 * core::NumSquares * I,
                Evaluator->getWinRate()[I],
                Evaluator->getDrawRate()[I]);

        Tasks.at(I)->setPhase(SelfplayPhase::Backpropagation);
    }

    SInfo->putBatchSizeStatistics(Tasks.size());
    SearchQueue->add(Tasks);

    return false;
}

void EvaluationWorker::prepareInfer([[ maybe_unused ]] std::size_t GPUId, [[ maybe_unused ]] const char* WeightPath) {
#if defined(EXECUTOR_ZERO)
    Infer = std::make_unique<infer::Zero>();
#elif defined(EXECUTOR_NOTHING)
    Infer = std::make_unique<infer::Nothing>();
#elif defined(EXECUTOR_RANDOM)
    Infer = std::make_unique<infer::Random>(0);
#elif defined(EXECUTOR_TRT)
    auto TRT = std::make_unique<infer::TensorRT>(GPUId, BatchSize, evaluate::preset::CustomFeaturesV1::size());
    TRT->load(WeightPath, true);
    Infer = std::move(TRT);
#endif
    Evaluator = std::make_unique<evaluate::Evaluator>(BatchSize, Infer.get());
}

void EvaluationWorker::allocate() {
#ifdef CUDA_ENABLED
    cudaMallocHost(
        &FeatureBitboards,
        BatchSize * evaluate::preset::CustomFeaturesV1::size() * sizeof(ml::FeatureBitboard));
#else
    FeatureBitboards = new ml::FeatureBitboard[BatchSize * evaluate::preset::CustomFeaturesV1::size()];
#endif
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
