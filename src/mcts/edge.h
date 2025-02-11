//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_EDGE_H
#define NSHOGI_ENGINE_MCTS_EDGE_H
#include <atomic>
#include <memory>
#include <cassert>

#include "pointer.h"
#include "../allocator/allocator.h"
#include <nshogi/core/types.h>


namespace nshogi {
namespace engine {
namespace mcts {

struct Node;

struct Edge {
 public:
    Edge(): Ready(false) {
    }

    Edge(const Edge& E) {
        // You must not call the copy constructor
        // unless the target is nullptr.
        assert(E.Target == nullptr);

        Probability = E.Probability;
        Move = E.Move;
    }

    Edge& operator=(const Edge& E) {
        // You must not call the copy constructor
        // unless the target is nullptr.
        assert(E.Target == nullptr);

        Probability = E.Probability;
        Move = E.Move;

        return *this;
    }

    void setProbability(float P) {
        Probability = P;
    }

    float getProbability() const {
        return Probability;
    }

    void setTarget(Pointer<Node>&& T) {
        assert(Target == nullptr);
        assert(Ready == false);

        Target = std::move(T);
        Ready.store(true, std::memory_order_release);
    }

    Node* getTarget() {
        if (Ready.load(std::memory_order_acquire)) {
            assert(Target != nullptr);
            return Target.get();
        }

        return nullptr;
    }

    Pointer<Node>&& getTargetWithOwner() {
        return std::move(Target);
    }

    void setMove(core::Move16 M) {
        Move = M;
    }

    core::Move16 getMove() const {
        return Move;
    }

 private:
    Pointer<Node> Target;
    float Probability;
    core::Move16 Move;
    std::atomic<bool> Ready;
};


} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EDGE_H
