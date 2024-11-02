#include "evaluateworker.h"
#include "../globalconfig.h"

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

#include <chrono>

#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
EvaluateWorker<Features>::EvaluateWorker(const Context* C, std::size_t GPUId, std::size_t BatchSize, EvaluationQueue<Features>* EQ, EvalCache* EC)
    : worker::Worker(true)
    , PContext(C)
    , BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , ECache(EC)
    , GPUId_(GPUId)
    , SequentialSkip(0) {
    PendingSideToMoves.reserve(BatchSize);
    PendingNodes.reserve(BatchSize);
    PendingFeatures.reserve(BatchSize);
    PendingHashes.reserve(BatchSize);

    spawnThread();
}

template <typename Features>
EvaluateWorker<Features>::~EvaluateWorker() {
#ifdef CUDA_ENABLED
    cudaFree(FeatureBitboards);
#else
    delete[] FeatureBitboards;
#endif
}

template <typename Features>
void EvaluateWorker<Features>::initializationTask() {
#ifdef CUDA_ENABLED
    cudaMallocHost(&FeatureBitboards, BatchSizeMax * Features::size() * sizeof(ml::FeatureBitboard));
#else
    FeatureBitboards = new ml::FeatureBitboard[BatchSizeMax * Features::size()];
#endif

#if defined(EXECUTOR_ZERO)
    Infer = std::make_unique<infer::Zero>();
#elif defined(EXECUTOR_NOTHING)
    Infer = std::make_unique<infer::Nothing>();
#elif defined(EXECUTOR_RANDOM)
    Infer = std::make_unique<infer::Random>(0);
#elif defined(EXECUTOR_TRT)
    auto TRT = std::make_unique<infer::TensorRT>(GPUId_, BatchSizeMax, global_config::FeatureType::size());
    TRT->load(PContext->getWeightPath(), true);
    TRT->resetGPU();
    Infer = std::move(TRT);
#endif
    Evaluator = std::make_unique<evaluate::Evaluator>(BatchSizeMax, Infer.get());
}

template <typename Features>
bool EvaluateWorker<Features>::doTask() {
    getBatch();
    const std::size_t BatchSize = PendingSideToMoves.size();

    if (BatchSize == 0) {
        std::this_thread::yield();
        // As the batch size is zero, there is no tasks to do, so
        // return false to notify this thread can be stopped.
        return false;
    }

    if (SequentialSkip <= SEQUENTIAL_SKIP_THRESHOLD && BatchSize < BatchSizeMax / 2) {
        ++SequentialSkip;
        std::this_thread::yield();
        // There are tasks to do so return true not to stop this worker.
        return true;
    }

    flattenFeatures(BatchSize);
    doInference(BatchSize);
    feedResults(BatchSize);

    PendingNodes.clear();
    PendingSideToMoves.clear();
    PendingFeatures.clear();
    PendingHashes.clear();
    SequentialSkip = 0;

    // There may be tasks (a next batch) to do so return true not to stop this worker.
    return true;
}

template <typename Features>
void EvaluateWorker<Features>::getBatch() {
    if (PendingSideToMoves.size() >= BatchSizeMax) {
        return;
    }

    auto Elements = EQueue->get(BatchSizeMax - PendingSideToMoves.size());

    auto SideToMoves = std::move(std::get<0>(Elements));
    auto Nodes = std::move(std::get<1>(Elements));
    auto FeatureStacks = std::move(std::get<2>(Elements));
    auto Hashes = std::move(std::get<3>(Elements));

    if (SideToMoves.size() == 0) {
        return;
    }

    std::move(SideToMoves.begin(), SideToMoves.end(), std::back_inserter(PendingSideToMoves));
    std::move(Nodes.begin(), Nodes.end(), std::back_inserter(PendingNodes));
    std::move(FeatureStacks.begin(), FeatureStacks.end(), std::back_inserter(PendingFeatures));
    std::move(Hashes.begin(), Hashes.end(), std::back_inserter(PendingHashes));
}

template <typename Features>
void EvaluateWorker<Features>::flattenFeatures(std::size_t BatchSize) {
    const std::size_t UnitSize = PendingFeatures[0].size();

    for (std::size_t I = 0; I < BatchSize; ++I) {
        std::memcpy(
            static_cast<void*>((ml::FeatureBitboard*)(FeatureBitboards) + I * UnitSize),
            PendingFeatures[I].data(),
            UnitSize * sizeof(ml::FeatureBitboard));
    }
}

template <typename Features>
void EvaluateWorker<Features>::doInference(std::size_t BatchSize) {
    Evaluator->computeBlocking(FeatureBitboards, BatchSize);
}

template <typename Features>
void EvaluateWorker<Features>::feedResults(std::size_t BatchSize) {
    for (std::size_t I = 0; I < BatchSize; ++I) {
        const float* Policy = Evaluator->getPolicy() + 27 * core::NumSquares * I;
        const float WinRate = *(Evaluator->getWinRate() + I);
        const float DrawRate = *(Evaluator->getDrawRate() + I);

        assert(WinRate >= 0.0f && WinRate <= 1.0f);
        assert(DrawRate >= 0.0f && DrawRate <= 1.0f);
        feedResult(PendingSideToMoves[I], PendingNodes[I], Policy, WinRate, DrawRate, PendingHashes[I]);
    }
}

template <typename Features>
void EvaluateWorker<Features>::feedResult(core::Color SideToMove, Node* N, const float* Policy, float WinRate, float DrawRate, uint64_t Hash) {
    const uint16_t NumChildren = N->getNumChildren();
    if (NumChildren == 1) {
        constexpr float P[] = { 1.0 };
        N->setEvaluation(P, WinRate, DrawRate);
    } else {
        for (uint16_t I = 0; I < NumChildren; ++I) {
            const std::size_t MoveIndex = ml::getMoveIndex(SideToMove, N->getEdge()[I].getMove());
            LegalPolicy[I] = Policy[MoveIndex];
        }
        ml::math::softmax_(LegalPolicy, NumChildren, 1.6f);
        N->setEvaluation(LegalPolicy, WinRate, DrawRate);
        N->sort();
    }

    N->updateAncestors(WinRate, DrawRate);

    if (ECache != nullptr) {
        ECache->store(Hash, NumChildren, LegalPolicy, WinRate, DrawRate);
    }
}

template class EvaluateWorker<evaluate::preset::SimpleFeatures>;
template class EvaluateWorker<evaluate::preset::CustomFeaturesV1>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
