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
EvaluateWorker<Features>::EvaluateWorker(std::size_t BatchSize, EvaluationQueue<Features>* EQ, evaluate::Evaluator* Ev)
    : worker::Worker(true)
    , BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , Evaluator(Ev)
    , SequentialSkip(0) {
    FeatureBitboards.resize(Features::size() * BatchSize);

    PendingSideToMoves.reserve(BatchSize);
    PendingNodes.reserve(BatchSize);
    PendingFeatures.reserve(BatchSize);

    // Worker = std::thread(&EvaluateWorker<Features>::mainLoop, this);
}

template <typename Features>
EvaluateWorker<Features>::~EvaluateWorker() {
}

// template <typename Features>
// void EvaluateWorker<Features>::start() {
//     IsRunnning.store(true, std::memory_order_release);
//     CV.notify_one();
// }

// template <typename Features>
// void EvaluateWorker<Features>::stop() {
//     IsRunnning.store(false, std::memory_order_release);
//
//     while (IsThreadWorking.load(std::memory_order_acquire)) {
//         // Busy loop until the thread stops.
//         std::this_thread::yield();
//     }
// }

// template <typename Features>
// void EvaluateWorker<Features>::mainLoop() {
//     while (true) {
//         std::unique_lock<std::mutex> Lock(Mutex);
//
//         if (!IsRunnning.load(std::memory_order_relaxed)) {
//             IsThreadWorking.store(false, std::memory_order_relaxed);
//         }
//
//         CV.wait(Lock, [this]() {
//             return IsRunnning.load(std::memory_order_relaxed)
//                     || IsExiting.load(std::memory_order_relaxed);
//         });
//
//         if (IsExiting.load(std::memory_order_relaxed)) {
//             return;
//         }
//
//         IsThreadWorking.store(true, std::memory_order_relaxed);
//
//         while (true) {
//         }
//     }
// }

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

    flattenFeatures(PendingFeatures);
    doInference(BatchSize);
    feedResults(PendingSideToMoves, PendingNodes);

    PendingNodes.clear();
    PendingSideToMoves.clear();
    PendingFeatures.clear();
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

    if (SideToMoves.size() == 0) {
        return;
    }

    std::move(SideToMoves.begin(), SideToMoves.end(), std::back_inserter(PendingSideToMoves));
    std::move(Nodes.begin(), Nodes.end(), std::back_inserter(PendingNodes));
    std::move(FeatureStacks.begin(), FeatureStacks.end(), std::back_inserter(PendingFeatures));
}

template <typename Features>
void EvaluateWorker<Features>::flattenFeatures(const std::vector<Features>& FSC) {
    for (std::size_t I = 0; I < FSC.size(); ++I) {
        std::memcpy(static_cast<void*>(FeatureBitboards.data() + I * FSC[I].size()),
                    FSC[I].data(),
                    FSC[I].size() * sizeof(ml::FeatureBitboard));
    }
}

template <typename Features>
void EvaluateWorker<Features>::doInference(std::size_t BatchSize) {
    Evaluator->computeBlocking(FeatureBitboards.data(), BatchSize);
}

template <typename Features>
void EvaluateWorker<Features>::feedResults(const std::vector<core::Color>& SideToMoves, const std::vector<Node*>& Nodes) {
    for (std::size_t I = 0; I < Nodes.size(); ++I) {
        const float* Policy = Evaluator->getPolicy() + 27 * core::NumSquares * I;
        const float WinRate = *(Evaluator->getWinRate() + I);
        const float DrawRate = *(Evaluator->getDrawRate() + I);

        feedResult(SideToMoves[I], Nodes[I], Policy, WinRate, DrawRate);
    }
}

template <typename Features>
void EvaluateWorker<Features>::feedResult(core::Color SideToMove, Node* N, const float* Policy, float WinRate, float DrawRate) {
    const uint16_t NumChildren = N->getNumChildren();
    for (uint16_t I = 0; I < NumChildren; ++I) {
        const std::size_t MoveIndex = ml::getMoveIndex(SideToMove, N->getEdge(I)->getMove());
        LegalPolicy[I] = Policy[MoveIndex];
    }

    ml::math::softmax_(LegalPolicy, NumChildren, 1.6f);
    N->setEvaluation(LegalPolicy, WinRate, DrawRate);

    N->sort();
    N->updateAncestors(WinRate, DrawRate);
}

template class EvaluateWorker<evaluate::preset::SimpleFeatures>;
template class EvaluateWorker<evaluate::preset::CustomFeaturesV1>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
