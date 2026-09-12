//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../allocator/default.h"
#include "../mcts/tree.h"

#include <gtest/gtest.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

#include <barrier>
#include <thread>
#include <vector>

namespace {

using namespace nshogi;
using namespace nshogi::engine;

class SearchTree : public testing::Test {
 protected:
    allocator::DefaultAllocator Allocator;
    mcts::GarbageCollector GC{1, &Allocator, &Allocator};
    mcts::Tree Tree{&GC, &Allocator, nullptr};
    core::State Initial = core::StateBuilder::getInitialState();

    mcts::Node* expandRoot(const core::State& State) {
        auto* Root = Tree.updateRoot(State, false);
        Root->expand(core::MoveGenerator::generateLegalMoves(State),
                     &Allocator);
        return Root;
    }

    mcts::Node* addChild(mcts::Node* Parent, uint16_t Index, uint32_t Visits,
                         int16_t SolvedPly = 0) {
        mcts::Pointer<mcts::Node> Child;
        auto* N = Child.malloc(&Allocator, Parent);
        for (uint32_t I = 0; I < Visits; ++I) {
            N->incrementVisits();
        }
        N->setPlyToTerminalSolved(SolvedPly);
        Parent->publishChild(&Parent->getEdge()[Index], std::move(Child));
        return N;
    }

    void expectState(const core::State& State) {
        EXPECT_EQ(Tree.getRootState()->getHash(), State.getHash());
        EXPECT_EQ(Tree.getRootState()->getPly(), State.getPly());
        EXPECT_EQ(Tree.getRootState()->getPly(false), State.getPly(false));
        EXPECT_EQ(Tree.getRoot()->getParent(), nullptr);
    }
};

TEST_F(SearchTree, OutOfOrderPublicationKeepsTheWholeRange) {
    auto* Root = expandRoot(Initial);
    EXPECT_EQ(Root->getExpandedEnd(), 0);
    addChild(Root, 5, 0);
    addChild(Root, 1, 0);
    EXPECT_EQ(Root->getExpandedEnd(), 6);
    addChild(Root, 9, 0);
    EXPECT_EQ(Root->getExpandedEnd(), 10);
}

TEST_F(SearchTree, ConcurrentPublicationIncludesEveryVisibleChild) {
    auto* Root = expandRoot(Initial);
    const uint16_t Count = Root->getNumChildren();
    std::vector<mcts::Pointer<mcts::Node>> Children(Count);
    for (auto& Child : Children) {
        ASSERT_NE(Child.malloc(&Allocator, Root), nullptr);
    }

    constexpr uint16_t NumThreads = 4;
    std::barrier Start(NumThreads + 1);
    std::vector<std::thread> Publishers;
    for (uint16_t T = 0; T < NumThreads; ++T) {
        Publishers.emplace_back([&, T] {
            Start.arrive_and_wait();
            for (uint16_t I = T; I < Count; I += NumThreads) {
                Root->publishChild(&Root->getEdge()[I],
                                   std::move(Children[I]));
            }
        });
    }
    Start.arrive_and_wait();
    for (uint16_t I = 0; I < Count; ++I) {
        while (Root->getEdge()[I].getTarget() == nullptr) {
            std::this_thread::yield();
        }
        // Acquiring the published pointer must also expose its scan bound.
        EXPECT_GT(Root->getExpandedEnd(), I);
    }
    for (auto& Publisher : Publishers) {
        Publisher.join();
    }
    EXPECT_EQ(Root->getExpandedEnd(), Count);
}

TEST_F(SearchTree, ReleasingEdgesResetsPublishedRange) {
    auto* Root = expandRoot(Initial);
    addChild(Root, 4, 0);
    GC.addGarbage(std::move(Root->getEdge()[4].getTargetWithOwner()));
    Root->releaseEdges(&Allocator);
    EXPECT_EQ(Root->getExpandedEnd(), 0);
    EXPECT_EQ(Root->getNumChildren(), 0);
    EXPECT_EQ(Root->getEdge(), nullptr);
    Root->expand(core::MoveGenerator::generateLegalMoves(Initial), &Allocator);
    addChild(Root, 1, 0);
    EXPECT_EQ(Root->getExpandedEnd(), 2);
}

} // namespace
