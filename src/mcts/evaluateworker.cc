#include "evaluateworker.h"
#include "../evaluate/preset.h"

#include <chrono>

#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace mcts {

namespace {

void cancelVirtualLoss(Node* N) {
    do {
        N->decrementVirtualLoss();
        N = N->getParent();
    } while (N != nullptr);
}

} // namespace

template <typename Features>
EvaluateWorker<Features>::EvaluateWorker(std::size_t BatchSize, EvaluationQueue<Features>* EQ, evaluate::Evaluator* Ev, EvalCache* EC)
    : worker::Worker(true)
    , BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , Evaluator(Ev)
    , ECache(EC)
    , SequentialSkip(0) {
    FeatureBitboards.resize(Features::size() * BatchSize);

    PendingSideToMoves.reserve(BatchSize);
    PendingNodes.reserve(BatchSize);
    PendingFeatures.reserve(BatchSize);
    PendingHashes.reserve(BatchSize);
}

template <typename Features>
EvaluateWorker<Features>::~EvaluateWorker() {
}

template <typename Features>
bool EvaluateWorker<Features>::doTask() {
    getBatch();
    const std::size_t BatchSize = PendingSideToMoves.size();

    if (BatchSize == 0) {
        std::this_thread::yield();
        return false;
    }

    if (SequentialSkip <= SEQUENTIAL_SKIP_THRESHOLD && BatchSize < BatchSizeMax / 2) {
        ++SequentialSkip;
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

    return true;
}

template <typename Features>
void EvaluateWorker<Features>::getBatch() {
    if (PendingSideToMoves.size() >= BatchSizeMax) {
        return;
    }

    auto Elements = std::move(EQueue->get(BatchSizeMax - PendingSideToMoves.size()));

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
    for (std::size_t I = 0; I < BatchSize; ++I) {
        std::memcpy(
            static_cast<void*>(FeatureBitboards.data() + I * PendingFeatures[I].size()),
            PendingFeatures[I].data(),
            PendingFeatures[I].size() * sizeof(ml::FeatureBitboard));
    }
}

template <typename Features>
void EvaluateWorker<Features>::doInference(std::size_t BatchSize) {
    Evaluator->computeBlocking(FeatureBitboards.data(), BatchSize);
}

template <typename Features>
void EvaluateWorker<Features>::feedResults(std::size_t BatchSize) {
    for (std::size_t I = 0; I < BatchSize; ++I) {
        const float* Policy = Evaluator->getPolicy() + 27 * core::NumSquares * I;
        const float WinRate = *(Evaluator->getWinRate() + I);
        const float DrawRate = *(Evaluator->getDrawRate() + I);

        feedResult(PendingSideToMoves[I], PendingNodes[I], Policy, WinRate, DrawRate, PendingHashes[I]);
    }
}

template <typename Features>
void EvaluateWorker<Features>::feedResult(core::Color SideToMove, Node* N, const float* Policy, float WinRate, float DrawRate, uint64_t Hash) {
    const uint16_t NumChildren = N->getNumChildren();
    for (uint16_t I = 0; I < NumChildren; ++I) {
        const std::size_t MoveIndex = ml::getMoveIndex(SideToMove, N->getEdge(I)->getMove());
        LegalPolicy[I] = Policy[MoveIndex];
    }

    ml::math::softmax_(LegalPolicy, NumChildren, 1.6f);
    N->setEvaluation(LegalPolicy, WinRate, DrawRate);
    N->sort();
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
