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

TEST_F(SearchTree, UnexpandedRootHasNoBestEdge) {
    EXPECT_EQ(Tree.updateRoot(Initial)->mostPromisingEdge(), nullptr);
}

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
                Root->publishChild(&Root->getEdge()[I], std::move(Children[I]));
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

TEST_F(SearchTree, UnsearchedMoveIsPreferredToProvenLoss) {
    auto* Root = expandRoot(Initial);
    addChild(Root, 0, 10, 3);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[1]);
}

TEST_F(SearchTree, UnevaluatedChildIsPreferredToProvenLoss) {
    auto* Root = expandRoot(Initial);
    addChild(Root, 0, 10, 3);
    addChild(Root, 1, 0);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[1]);
}

TEST_F(SearchTree, BestMoveCountsOnlyRealVisitsOfUnsolvedChildren) {
    auto* Root = expandRoot(Initial);
    addChild(Root, 0, 100, 3);
    auto* Pending = addChild(Root, 1, 1);
    for (int I = 0; I < 10; ++I) {
        Pending->incrementVirtualLoss();
    }
    addChild(Root, 2, 3);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[2]);
    for (int I = 0; I < 10; ++I) {
        Pending->decrementVirtualLoss();
    }
}

TEST_F(SearchTree, ProvenWinAndSolverMoveOverrideVisits) {
    auto* Root = expandRoot(Initial);
    addChild(Root, 0, 100);
    addChild(Root, 2, 1, -3);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[2]);
    Root->setSolverResult(Root->getEdge()[1].getMove());
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[1]);
}

TEST_F(SearchTree, EqualVisitsKeepPriorOrder) {
    auto* Root = expandRoot(Initial);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[0]);
    addChild(Root, 0, 3);
    addChild(Root, 1, 3);
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[0]);
}

TEST_F(SearchTree, AllProvenLossesStillReturnAMove) {
    auto* Root = expandRoot(Initial);
    for (uint16_t I = 0; I < Root->getNumChildren(); ++I) {
        addChild(Root, I, 1, 3);
    }
    EXPECT_EQ(Root->mostPromisingEdge(), &Root->getEdge()[0]);
}

TEST_F(SearchTree, RewindReplacesRootDespiteRetainedLastMove) {
    auto State = io::sfen::StateBuilder::newState("startpos moves 7g7f");
    Tree.updateRoot(State)->incrementVisits();
    State.undoMove();
    Tree.updateRoot(State);
    expectState(State);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 0U);
}

TEST_F(SearchTree, ReusesChildrenWithAndWithoutSfenPlyOffset) {
    for (const char* Sfen : {"startpos", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
                                         "PPPPPPPPP/1B5R1/LNSGKGSNL b - 100"}) {
        SCOPED_TRACE(Sfen);
        auto State = io::sfen::StateBuilder::newState(Sfen);
        auto* Root = expandRoot(State);
        auto* Child = addChild(Root, 0, 17);
        State.doMove(State.getMove32FromMove16(Root->getEdge()[0].getMove()));
        EXPECT_EQ(Tree.updateRoot(State), Child);
        expectState(State);
        EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 17U);
    }
}

TEST_F(SearchTree, RepeatedSfenRootWithoutHistoryIsReused) {
    auto State = io::sfen::StateBuilder::newState(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 5000");
    auto* Root = expandRoot(State);
    Root->incrementVisits();
    EXPECT_EQ(Tree.updateRoot(State), Root);
    expectState(State);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 1U);
}

TEST_F(SearchTree, DifferentMoveHistoryReplacesRoot) {
    auto First = io::sfen::StateBuilder::newState("startpos moves 7g7f");
    auto Other = io::sfen::StateBuilder::newState("startpos moves 2g2f");
    Tree.updateRoot(First)->incrementVisits();
    Tree.updateRoot(Other);
    expectState(Other);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 0U);
}

TEST_F(SearchTree, ReusesGrandchildAndDetachesItsParent) {
    auto State = Initial.clone();
    auto* Root = expandRoot(State);
    auto* Child = addChild(Root, 0, 0);
    State.doMove(State.getMove32FromMove16(Root->getEdge()[0].getMove()));
    Child->expand(core::MoveGenerator::generateLegalMoves(State), &Allocator);
    auto* Grandchild = addChild(Child, 0, 7);
    EXPECT_EQ(Child->getExpandedEnd(), 1);
    EXPECT_EQ(Tree.updateRoot(State), Child);
    EXPECT_EQ(Child->getExpandedEnd(), 1);
    State.doMove(State.getMove32FromMove16(Child->getEdge()[0].getMove()));
    EXPECT_EQ(Tree.updateRoot(State), Grandchild);
    expectState(State);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 7U);
}

TEST_F(SearchTree, MissingChildCreatesRootAtRequestedPosition) {
    auto State = Initial.clone();
    auto* Root = expandRoot(State);
    State.doMove(State.getMove32FromMove16(Root->getEdge()[0].getMove()));
    Tree.updateRoot(State);
    expectState(State);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 0U);
}

TEST_F(SearchTree, RepetitionChildIsReplacedWithFreshRoot) {
    auto State = Initial.clone();
    auto* Root = expandRoot(State);
    addChild(Root, 0, 7)
        ->setRepetitionStatus(core::RepetitionStatus::Repetition);
    State.doMove(State.getMove32FromMove16(Root->getEdge()[0].getMove()));
    Tree.updateRoot(State);
    expectState(State);
    EXPECT_EQ(Tree.getRoot()->getVisitsAndVirtualLoss(), 0U);
}

} // namespace
