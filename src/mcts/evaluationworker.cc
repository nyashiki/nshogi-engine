//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evaluationworker.h"
#include "../globalconfig.h"

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

#include "../math/math.h"
#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace mcts {

EvaluationWorker::EvaluationWorker(const Context* C, std::size_t ThreadId,
                                   std::size_t GPUId, std::size_t BatchSize,
                                   EvaluationQueue* EQ, FeedQueue* FQ,
                                   Statistics* Stat)
    : worker::Worker(true)
    , PContext(C)
    , MyThreadId(ThreadId)
    , BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , FQueue(FQ)
    , GPUId_(GPUId)
    , BatchCount(0)
    , PendingSideToMoves(nullptr)
    , PendingNodes(nullptr)
    , PendingHashes(nullptr)
    , PStat(Stat) {

    spawnThread();
}

EvaluationWorker::~EvaluationWorker() {
    if (Evaluator != nullptr) {
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingSideToMoves),
                              BatchSizeMax * sizeof(core::Color));
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingNodes),
                              BatchSizeMax * sizeof(Node*));
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingHashes),
                              BatchSizeMax * sizeof(uint64_t));
    }
}

void EvaluationWorker::initializationTask() {
#if defined(EXECUTOR_ZERO)
    Infer = std::make_unique<infer::Zero>();
#elif defined(EXECUTOR_NOTHING)
    Infer = std::make_unique<infer::Nothing>();
#elif defined(EXECUTOR_RANDOM)
    Infer = std::make_unique<infer::Random>(0);
#elif defined(EXECUTOR_TRT)
    auto TRT = std::make_unique<infer::TensorRT>(
        GPUId_, BatchSizeMax, global_config::FeatureType::size());
    TRT->load(PContext->getWeightPath(), true);
    TRT->resetGPU();
    Infer = std::move(TRT);
#endif

    Evaluator = std::make_unique<evaluate::Evaluator>(
        MyThreadId, global_config::FeatureType::size(), BatchSizeMax,
        Infer.get());

    PendingSideToMoves =
        static_cast<core::Color*>(Evaluator->allocateMemoryByNumaIfAvailable(
            BatchSizeMax * sizeof(core::Color)));
    PendingNodes =
        static_cast<Node**>(Evaluator->allocateMemoryByNumaIfAvailable(
            BatchSizeMax * sizeof(Node*)));
    PendingHashes =
        static_cast<uint64_t*>(Evaluator->allocateMemoryByNumaIfAvailable(
            BatchSizeMax * sizeof(uint64_t)));
}

bool EvaluationWorker::doTask() {
    getBatch();

    if (BatchCount == 0) {
        // As the batch size is zero, there is no tasks to do, so
        // return false to notify this thread can be stopped.
        return EQueue->count() > 0;
    }

    auto B = doInference();
    addToFeedQueue(std::move(B));

    BatchCount = 0;

    // There may be tasks (a next batch) to do so return true not to stop this
    // worker.
    return true;
}

void EvaluationWorker::getBatch() {
    if (BatchCount >= BatchSizeMax) {
        return;
    }

    auto Elements = EQueue->get(BatchSizeMax - BatchCount);

    auto SideToMoves = std::move(std::get<0>(Elements));

    if (SideToMoves.size() == 0) {
        return;
    }

    auto Nodes = std::move(std::get<1>(Elements));
    auto FeatureStacks = std::move(std::get<2>(Elements));
    auto Hashes = std::move(std::get<3>(Elements));

    for (std::size_t I = 0; I < SideToMoves.size(); ++I) {
        PendingSideToMoves[BatchCount] = SideToMoves[I];
        PendingNodes[BatchCount] = Nodes[I];
        PendingHashes[BatchCount] = Hashes[I];

        std::memcpy(
            static_cast<void*>(Evaluator->getFeatureBitboards() +
                               BatchCount * global_config::FeatureType::size()),
            FeatureStacks[I].data(),
            global_config::FeatureType::size() * sizeof(ml::FeatureBitboard));

        ++BatchCount;
    }
}

std::unique_ptr<Batch> EvaluationWorker::doInference() {
    // Start inference.
    Evaluator->computeNonBlocking(BatchCount);

    // Prepare the batch while the inference is running.
    std::unique_ptr<Node*[]> Nodes =
        std::unique_ptr<Node*[]>(new Node*[BatchCount]);
    std::unique_ptr<uint64_t[]> Hashes =
        std::unique_ptr<uint64_t[]>(new uint64_t[BatchCount]);
    std::unique_ptr<uint32_t[]> PolicyOffsets =
        std::unique_ptr<uint32_t[]>(new uint32_t[BatchCount + 1]);
    std::unique_ptr<float[]> WinRates =
        std::unique_ptr<float[]>(new float[BatchCount]);
    std::unique_ptr<float[]> DrawRates =
        std::unique_ptr<float[]>(new float[BatchCount]);

    // Copy input.
    std::memcpy(Nodes.get(), PendingNodes, BatchCount * sizeof(Node*));
    std::memcpy(Hashes.get(), PendingHashes, BatchCount * sizeof(uint64_t));

    uint32_t NumLegalMoves = 0;
    for (std::size_t I = 0; I < BatchCount; ++I) {
        PolicyOffsets[I] = NumLegalMoves;
        NumLegalMoves += PendingNodes[I]->getNumChildren();
    }
    PolicyOffsets[BatchCount] = NumLegalMoves;
    std::unique_ptr<float[]> LegalPolicies =
        std::unique_ptr<float[]>(new float[NumLegalMoves]);

    PStat->incrementEvaluationCount();
    PStat->addBatchSizeAccumulated(BatchCount);

    // Await inference.
    Evaluator->await();

    // Copy output: gather the logits of the legal moves directly out
    // of the evaluator's output buffer. The feed worker consumes
    // nothing else of the policy head, so the raw policy (27 x 81
    // floats per position) is not copied around.
    const float* RawPolicy = Evaluator->getPolicy();
    for (std::size_t I = 0; I < BatchCount; ++I) {
        const Node* N = PendingNodes[I];
        const uint16_t NumChildren = N->getNumChildren();
        const float* Src = RawPolicy + 27 * core::NumSquares * I;
        float* Dst = LegalPolicies.get() + PolicyOffsets[I];
        for (uint16_t J = 0; J < NumChildren; ++J) {
            const std::size_t MoveIndex =
                ml::getMoveIndex<global_config::ChannelsFirst>(
                    PendingSideToMoves[I], N->getEdge()[J].getMove());
            Dst[J] = Src[MoveIndex];
        }
    }
    std::memcpy(WinRates.get(), Evaluator->getWinRate(),
                BatchCount * sizeof(float));
    std::memcpy(DrawRates.get(), Evaluator->getDrawRate(),
                BatchCount * sizeof(float));

    std::unique_ptr<Batch> B = std::make_unique<Batch>(
        BatchCount, std::move(Nodes), std::move(Hashes),
        std::move(PolicyOffsets), std::move(LegalPolicies),
        std::move(WinRates), std::move(DrawRates));

    return B;
}

void EvaluationWorker::addToFeedQueue(std::unique_ptr<Batch>&& B) {
    FQueue->add(std::move(B));
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
