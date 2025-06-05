//
// Copyright (c) 2025 @nyashiki
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

template <typename Features>
EvaluationWorker<Features>::EvaluationWorker(
    const Context* C, std::size_t ThreadId, std::size_t GPUId,
    std::size_t BatchSize, EvaluationQueue* EQ, EvalCache* EC, Statistics* Stat)
    : worker::Worker(true)
    , PContext(C)
    , MyThreadId(ThreadId)
    , BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , ECache(EC)
    , GPUId_(GPUId)
    , BatchCount(0)
    , PendingSideToMoves(nullptr)
    , PendingNodes(nullptr)
    , PendingHashes(nullptr) {

    spawnThread();
}

template <typename Features>
EvaluationWorker<Features>::~EvaluationWorker() {
    if (Evaluator != nullptr) {
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingSideToMoves),
                              BatchSizeMax * sizeof(core::Color));
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingNodes),
                              BatchSizeMax * sizeof(Node*));
        Evaluator->freeMemory(reinterpret_cast<void**>(&PendingHashes),
                              BatchSizeMax * sizeof(uint64_t));
    }
}

template <typename Features>
void EvaluationWorker<Features>::initializationTask() {
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
        MyThreadId, Features::size(), BatchSizeMax, Infer.get());

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

template <typename Features>
bool EvaluationWorker<Features>::doTask() {
    getBatch();

    if (BatchCount == 0) {
        std::this_thread::yield();
        // As the batch size is zero, there is no tasks to do, so
        // return false to notify this thread can be stopped.
        return isRunning() || EQueue->count() > 0;
    }

    doInference();
    feedResults();

    BatchCount = 0;

    // There may be tasks (a next batch) to do so return true not to stop this
    // worker.
    return true;
}

template <typename Features>
void EvaluationWorker<Features>::getBatch() {
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

        std::memcpy(static_cast<void*>(Evaluator->getFeatureBitboards() +
                                       BatchCount * Features::size()),
                    FeatureStacks[I].data(),
                    Features::size() * sizeof(ml::FeatureBitboard));

        ++BatchCount;
    }
}

template <typename Features>
void EvaluationWorker<Features>::doInference() {
    Evaluator->computeBlocking(BatchCount);
    PStat->incrementEvaluationCount();
    PStat->addBatchSizeAccumulated(BatchCount);
}

template <typename Features>
void EvaluationWorker<Features>::feedResults() {
    for (std::size_t I = 0; I < BatchCount; ++I) {
        const float* Policy =
            Evaluator->getPolicy() + 27 * core::NumSquares * I;
        const float WinRate = *(Evaluator->getWinRate() + I);
        const float DrawRate = *(Evaluator->getDrawRate() + I);

        feedResult(PendingSideToMoves[I], PendingNodes[I], Policy, WinRate,
                   DrawRate, PendingHashes[I]);
    }
}

template <typename Features>
void EvaluationWorker<Features>::feedResult(core::Color SideToMove, Node* N,
                                            const float* Policy, float WinRate,
                                            float DrawRate, uint64_t Hash) {
    bool NaNFound = false;
    if (PContext->isNaNFallbackEnabled()) {
        if (math::isnan_(WinRate)) {
            std::cerr << "WINRATE NAN FOUND" << std::endl;
            NaNFound = true;
            const Node* Parent = N->getParent();
            if (Parent == nullptr) {
                WinRate = 0.5f;
            } else {
                const double ParentWinRate =
                    Parent->getWinRateAccumulated() /
                    (Parent->getVisitsAndVirtualLoss() & Node::VisitMask);
                WinRate = (float)(1.0 - ParentWinRate);
            }
        }
        if (math::isnan_(DrawRate)) {
            std::cerr << "DRAWRATE NAN FOUND" << std::endl;
            NaNFound = true;
            const Node* Parent = N->getParent();
            if (Parent == nullptr) {
                DrawRate = 0.0f;
            } else {
                const double ParentDrawRate =
                    Parent->getDrawRateAccumulated() /
                    (Parent->getVisitsAndVirtualLoss() & Node::VisitMask);
                DrawRate = (float)ParentDrawRate;
            }
        }
    }

    assert(WinRate >= 0.0f && WinRate <= 1.0f);
    assert(DrawRate >= 0.0f && DrawRate <= 1.0f);

    const uint16_t NumChildren = N->getNumChildren();
    if (NumChildren == 1) {
        constexpr float P[] = {1.0};
        N->setEvaluation(P, WinRate, DrawRate);
    } else {
        if (PContext->isNaNFallbackEnabled()) {
            for (uint16_t I = 0; I < NumChildren; ++I) {
                const std::size_t MoveIndex =
                    ml::getMoveIndex(SideToMove, N->getEdge()[I].getMove());
                LegalPolicy[I] = Policy[MoveIndex];
                if (math::isnan_(LegalPolicy[I])) {
                    NaNFound = true;
                    for (uint16_t J = 0; J < NumChildren; ++J) {
                        LegalPolicy[J] = 1.0f;
                    }
                    break;
                }
            }
        } else {
            for (uint16_t I = 0; I < NumChildren; ++I) {
                const std::size_t MoveIndex =
                    ml::getMoveIndex(SideToMove, N->getEdge()[I].getMove());
                LegalPolicy[I] = Policy[MoveIndex];
            }
        }
        ml::math::softmax_(LegalPolicy, NumChildren, 1.6f);
        N->setEvaluation(LegalPolicy, WinRate, DrawRate);
        N->sort();
    }

    N->updateAncestors(WinRate, DrawRate);

    if (ECache != nullptr && !NaNFound) {
        ECache->store(Hash, NumChildren, LegalPolicy, WinRate, DrawRate);
    }
}

template class EvaluationWorker<global_config::FeatureType>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
