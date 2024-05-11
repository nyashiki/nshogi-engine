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
    : BatchSizeMax(BatchSize)
    , EQueue(EQ)
    , Evaluator(Ev)
    , IsRunnning(false)
    , IsThreadWorking(false)
    , IsExiting(false) {
    FeatureBitboards.resize(Features::size() * BatchSize);

    Worker = std::thread(&EvaluateWorker<Features>::mainLoop, this);
}

template <typename Features>
EvaluateWorker<Features>::~EvaluateWorker() {
    IsRunnning.store(false);
    IsExiting.store(true);
    CV.notify_one();

    Worker.join();
}

template <typename Features>
void EvaluateWorker<Features>::start() {
    IsRunnning.store(true, std::memory_order_release);
    CV.notify_one();
}

template <typename Features>
void EvaluateWorker<Features>::stop() {
    IsRunnning.store(false, std::memory_order_release);

    while (IsThreadWorking.load(std::memory_order_acquire)) {
        // Busy loop until the thread stops.
        std::this_thread::yield();
    }
}

template <typename Features>
void EvaluateWorker<Features>::mainLoop() {
    while (true) {
        std::unique_lock<std::mutex> Lock(Mutex);

        if (!IsRunnning.load(std::memory_order_relaxed)) {
            IsThreadWorking.store(false, std::memory_order_relaxed);
        }

        CV.wait(Lock, [this]() {
            return IsRunnning.load(std::memory_order_relaxed)
                    || IsExiting.load(std::memory_order_relaxed);
        });

        if (IsExiting.load(std::memory_order_relaxed)) {
            return;
        }

        IsThreadWorking.store(true, std::memory_order_relaxed);

        while (true) {
            auto Elements = std::move(EQueue->get(BatchSizeMax));

            const auto SideToMoves = std::move(std::get<0>(Elements));
            const auto Nodes = std::move(std::get<1>(Elements));

            const std::size_t BatchSize = SideToMoves.size();

            // if (!IsRunnning.load(std::memory_order_relaxed)) {
            //     if (BatchSize == 0) {
            //         break;
            //     }

            //     for (const auto& Node : Nodes) {
            //         cancelVirtualLoss(Node);
            //     }

            //     continue;
            // }

            if (BatchSize == 0) {
                if (!IsRunnning.load(std::memory_order_relaxed)) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            const auto FeatureStacks = std::move(std::get<2>(Elements));

            flattenFeatures(FeatureStacks);
            doInference(BatchSize);
            feedResults(SideToMoves, Nodes);
        }
    }
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
