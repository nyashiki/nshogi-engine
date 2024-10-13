#ifndef NSHOGI_ENGINE_MCTS_TREE_H
#define NSHOGI_ENGINE_MCTS_TREE_H


#include "node.h"
#include "edge.h"
#include "garbagecollector.h"
#include "../logger/logger.h"

#include <memory>
#include <nshogi/core/state.h>


namespace nshogi {
namespace engine {
namespace mcts {

class Tree {
 public:
    Tree(GarbageCollector* GCollector, logger::Logger* Logger);

    Node* updateRoot(const nshogi::core::State& State, bool ReUse = true);

    Node* getRoot() const;
    const core::State* getRootState() const;

 private:
    Node* createNewRoot(const nshogi::core::State& State);

    std::unique_ptr<Node> Root;
    std::unique_ptr<core::State> RootState;
    GarbageCollector* GC;
    logger::Logger* PLogger;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_TREE_H
