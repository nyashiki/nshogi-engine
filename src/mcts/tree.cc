#include "tree.h"
#include "garbagecollector.h"
#include <cstdint>
#include <memory>
#include <nshogi/io/sfen.h>


namespace nshogi {
namespace engine {
namespace mcts {

Tree::Tree(GarbageCollector* GCollector, logger::Logger* Logger)
    : GC(GCollector)
    , PLogger(Logger) {
}

Node* Tree::updateRoot(const nshogi::core::State& State, bool ReUse) {
    if (Root == nullptr || !ReUse) {
        return createNewRoot(State);
    }

    if (!RootState->getInitialPosition().equals(State.getInitialPosition())) {
        return createNewRoot(State);
    }

    uint16_t Ply = 0;

    for (; Ply < RootState->getPly(); ++Ply) {
        if (RootState->getHistoryMove(Ply) != State.getHistoryMove(Ply)) {
            return createNewRoot(State);
        }
    }

    std::vector<std::unique_ptr<Node>> Garbages;
    for (; Ply < State.getPly(); ++Ply) {
        const auto Move = State.getHistoryMove(Ply);
        const auto Move16 = core::Move16(Move);

        bool IsFound = false;

        for (uint16_t I = 0; I < Root->getNumChildren(); ++I) {
            Edge* E = Root->getEdge(I);

            if (E->getMove() == Move16) {
                IsFound = true;
                std::unique_ptr<Node> NextRoot = E->getTargetWithOwner();

                if (NextRoot == nullptr) {
                    return createNewRoot(State);
                }

                NextRoot->resetParent(nullptr);
                Garbages.emplace_back(std::move(Root));

                Root = std::move(NextRoot);
                RootState->doMove(Move);
                break;
            }
        }

        if (!IsFound || Root == nullptr) {
            GC->addGarbages(std::move(Garbages));
            return createNewRoot(State);
        }
    }

    PLogger->printLog("Existing node has been found.");

    if (Root->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
        PLogger->printLog("But it has repetition so create a new one.");
        GC->addGarbages(std::move(Garbages));
        return createNewRoot(State);
    }

    assert(Root->getParent() == nullptr);
    GC->addGarbages(std::move(Garbages));
    return Root.get();
}

Node* Tree::getRoot() const {
    return Root.get();
}

const core::State* Tree::getRootState() const {
    return RootState.get();
}

Node* Tree::createNewRoot(const nshogi::core::State& State) {
    if (PLogger != nullptr) {
        PLogger->printLog("Creating a new root.");
    }

    GC->addGarbage(std::move(Root));
    if (PLogger != nullptr) {
        PLogger->printLog("Throwed the previous root away.");
    }

    Root = std::make_unique<Node>(nullptr);

    if (Root == nullptr) {
        throw std::runtime_error("Failed to allocate a root node.");
    }

    if (PLogger != nullptr) {
        PLogger->printLog("New root is now allocated.");
    }

    RootState = std::make_unique<core::State>(State.clone());

    if (RootState == nullptr) {
        throw std::runtime_error("Failed to allocate a root state.");
    }

    if (PLogger != nullptr) {
        PLogger->printLog("RootState is set.");
    }

    return Root.get();
}


} // namespace mcts
} // namespace engine
} // namespace nshogi
