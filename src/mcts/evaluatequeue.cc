#include "evaluatequeue.h"

#include "../evaluate/preset.h"

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
EvaluationQueue<Features>::EvaluationQueue(std::size_t MaxSize)
    : MaxQueueSize(MaxSize) {
}

template <typename Features>
void EvaluationQueue<Features>::add(const core::State& State, const core::StateConfig& Config, Node* N) {
    Features FSC(State, Config);

    std::unique_lock<std::mutex> Lock(Mutex);

    CV.wait(Lock, [this]() {
        return Queue.size() < MaxQueueSize;
    });

    Queue.emplace(State.getSideToMove(), N, std::move(FSC), State.getHash(), N->getNumChildren());
}

template <typename Features>
auto EvaluationQueue<Features>::get(std::size_t NumElements) -> std::tuple<std::vector<core::Color>, std::vector<Node*>, std::vector<Features>, std::vector<uint64_t>, std::vector<uint16_t>> {
    std::vector<core::Color> SideToMoves;
    std::vector<Node*> Nodes;
    std::vector<Features> FeatureStacks;
    std::vector<uint64_t> Hashes;
    std::vector<uint16_t> NumMoves;

    SideToMoves.reserve(NumElements);
    Nodes.reserve(NumElements);
    FeatureStacks.reserve(NumElements);
    Hashes.reserve(NumElements);
    NumMoves.reserve(NumElements);

    {
        std::lock_guard<std::mutex> Lock(Mutex);
        while (!Queue.empty() && NumElements--) {
            auto Element = std::move(Queue.front());
            Queue.pop();

            SideToMoves.emplace_back(std::get<0>(Element));
            Nodes.emplace_back(std::get<1>(Element));
            FeatureStacks.emplace_back(std::move(std::get<2>(Element)));
            Hashes.emplace_back(std::get<3>(Element));
            NumMoves.emplace_back(std::get<4>(Element));
        }
    }

    CV.notify_all();

    return std::make_tuple(
            std::move(SideToMoves),
            std::move(Nodes),
            std::move(FeatureStacks),
            std::move(Hashes),
            std::move(NumMoves));
}

template class EvaluationQueue<evaluate::preset::SimpleFeatures>;
template class EvaluationQueue<evaluate::preset::CustomFeaturesV1>;


} // namespace mcts
} // namespace engine
} // namespace nshogi
