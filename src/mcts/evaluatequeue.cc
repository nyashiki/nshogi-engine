#include "evaluatequeue.h"

namespace nshogi {
namespace engine {
namespace mcts {


template <typename Features>
void EvaluationQueue<Features>::add(const core::State& State, const core::StateConfig& Config, Node* N) {
    Features FSC(State, Config);

    std::lock_guard<std::mutex> Lock(Mutex);
    Queue.emplace(State.getSideToMove(), N, std::move(FSC));
}

template <typename Features>
auto EvaluationQueue<Features>::get(std::size_t NumElements) -> std::tuple<std::vector<core::Color>, std::vector<Node*>, std::vector<Features>> {
    std::vector<core::Color> SideToMoves;
    std::vector<Node*> Nodes;
    std::vector<Features> FeatureStacks;

    SideToMoves.reserve(NumElements);
    Nodes.reserve(NumElements);
    FeatureStacks.reserve(NumElements);

    std::lock_guard<std::mutex> Lock(Mutex);
    while (!Queue.Empty && NumElements--) {
        const auto&& Element = Queue.front();
        Queue.pop();

        SideToMoves.emplace_back(std::get<0>(Element));
        Nodes.emplace_back(std::get<1>(Element));
        FeatureStacks.emplace_back(std::move(std::get<2>(Element)));
    }

    return std::make_tuple(std::move(SideToMoves), std::move(Nodes), std::move(FeatureStacks));
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
