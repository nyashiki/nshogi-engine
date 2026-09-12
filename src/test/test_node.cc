//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../allocator/default.h"
#include "../mcts/node.h"

#include <gtest/gtest.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>

#include <atomic>
#include <thread>
#include <vector>

namespace {

using namespace nshogi;
using namespace nshogi::engine;

// The relaxed flags only schedule the interleaving. In particular, they must
// not provide the acquire that the Node operation under test is responsible
// for. Check the snapshots after joining, without assertions in the readers.
TEST(NodePublication, VirtualLossAcquiresEvaluation) {
    allocator::DefaultAllocator Allocator;
    const auto State = core::StateBuilder::getInitialState();
    const auto Moves = core::MoveGenerator::generateLegalMoves(State);
    const std::vector<float> Policy(Moves.size(), 1.0f / (float)Moves.size());
    mcts::Node N(nullptr);
    N.incrementVirtualLoss();
    std::atomic<bool> Start{false};
    uint16_t Count = 0;
    float WinRate = 0, DrawRate = 0, Prior = 0;
    core::RepetitionStatus Repetition{};
    std::thread Writer([&]() {
        while (!Start.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }
        N.expand(Moves, &Allocator);
        N.setRepetitionStatus(core::RepetitionStatus::Repetition);
        N.setEvaluation(Policy.data(), 0.75f, 0.25f);
        N.sort();
        N.updateAncestors(0.75f, 0.25f);
    });
    std::thread Reader([&]() {
        Start.store(true, std::memory_order_relaxed);
        for (;;) {
            const uint64_t Previous = N.incrementVirtualLoss();
            if ((Previous & mcts::Node::VisitMask) != 0) {
                Count = N.getNumChildren();
                WinRate = N.getWinRatePredicted();
                DrawRate = N.getDrawRatePredicted();
                Repetition = N.getRepetitionStatus();
                Prior = N.getEdge()[0].getProbability();
                N.decrementVirtualLoss();
                break;
            }
            N.decrementVirtualLoss();
            std::this_thread::yield();
        }
    });
    Writer.join();
    Reader.join();
    EXPECT_EQ(Count, Moves.size());
    EXPECT_EQ(WinRate, 0.75f);
    EXPECT_EQ(DrawRate, 0.25f);
    EXPECT_EQ(Repetition, core::RepetitionStatus::Repetition);
    EXPECT_EQ(Prior, Policy[0]);
    EXPECT_EQ(N.getVisitsAndVirtualLoss(), 1U);
    N.releaseEdges(&Allocator);
}

TEST(NodePublication, VirtualLossAcquiresCancelledExpansion) {
    allocator::DefaultAllocator Allocator;
    const auto State = core::StateBuilder::getInitialState();
    const auto Moves = core::MoveGenerator::generateLegalMoves(State);
    mcts::Node N(nullptr);
    N.incrementVirtualLoss();
    std::atomic<bool> Start{false};
    uint16_t Count = 1;
    bool Empty = false;
    std::thread Writer([&]() {
        while (!Start.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }
        N.expand(Moves, &Allocator);
        // EvaluationQueue::add() failed: roll back before releasing ownership.
        N.releaseEdges(&Allocator);
        N.decrementVirtualLoss();
    });
    std::thread Reader([&]() {
        Start.store(true, std::memory_order_relaxed);
        while (N.incrementVirtualLoss() != 0) {
            N.decrementVirtualLoss();
            std::this_thread::yield();
        }
        // A zero return value acquires the cancelled expansion too, even
        // though no completed visit has ever been published.
        Count = N.getNumChildren();
        Empty = N.getEdge() == nullptr;
        N.expand(Moves, &Allocator);
        N.setEvaluation(nullptr, 0.5f, 0.0f);
        N.updateAncestors(0.5f, 0.0f);
    });
    Writer.join();
    Reader.join();
    EXPECT_EQ(Count, 0);
    EXPECT_TRUE(Empty);
    EXPECT_EQ(N.getNumChildren(), Moves.size());
    EXPECT_EQ(N.getVisitsAndVirtualLoss(), 1U);
    N.releaseEdges(&Allocator);
}

TEST(NodePublication, InterveningRMWPreservesEvaluationPublication) {
    mcts::Node N(nullptr);
    N.incrementVirtualLoss();
    std::atomic<int> Phase{0};
    float WinRate = 0, DrawRate = 0;
    std::thread Writer([&]() {
        while (Phase.load(std::memory_order_relaxed) == 0) {
            std::this_thread::yield();
        }
        N.setEvaluation(nullptr, 0.75f, 0.25f);
        N.updateAncestors(0.75f, 0.25f);
        Phase.store(2, std::memory_order_relaxed);
    });
    std::thread Relay([&]() {
        while (Phase.load(std::memory_order_relaxed) != 2) {
            std::this_thread::yield();
        }
        // A release-only RMW by another thread must not break the release
        // sequence headed by the evaluation's visit increment.
        N.incrementVisits();
        Phase.store(3, std::memory_order_relaxed);
    });
    std::thread Reader([&]() {
        Phase.store(1, std::memory_order_relaxed);
        while (Phase.load(std::memory_order_relaxed) != 3) {
            std::this_thread::yield();
        }
        N.incrementVirtualLoss();
        WinRate = N.getWinRatePredicted();
        DrawRate = N.getDrawRatePredicted();
        N.decrementVirtualLoss();
    });
    Writer.join();
    Relay.join();
    Reader.join();
    EXPECT_EQ(WinRate, 0.75f);
    EXPECT_EQ(DrawRate, 0.25f);
    EXPECT_EQ(N.getVisitsAndVirtualLoss(), 2U);
}

TEST(NodePublication, ConcurrentBackupsPreserveCountsAndScores) {
    mcts::Node Root(nullptr);
    mcts::Node Leaf(&Root);
    constexpr int NumThreads = 8, Iterations = 1000;
    std::vector<std::thread> Threads;
    for (int I = 0; I < NumThreads; ++I) {
        Threads.emplace_back([&]() {
            for (int J = 0; J < Iterations; ++J) {
                Root.incrementVirtualLoss();
                Leaf.incrementVirtualLoss();
                Leaf.updateAncestors(0.75f, 0.5f);
            }
        });
    }
    for (auto& Thread : Threads) {
        Thread.join();
    }
    constexpr uint64_t Visits = NumThreads * Iterations;
    EXPECT_EQ(Leaf.getVisitsAndVirtualLoss(), Visits);
    EXPECT_EQ(Root.getVisitsAndVirtualLoss(), Visits);
    EXPECT_EQ(Leaf.getExpectedScoreAccumulated(), 0.625 * Visits);
    EXPECT_EQ(Root.getExpectedScoreAccumulated(), 0.375 * Visits);
    EXPECT_EQ(Leaf.getDrawRateAccumulated(), 0.5 * Visits);
    EXPECT_EQ(Root.getDrawRateAccumulated(), 0.5 * Visits);
    EXPECT_EQ(sizeof(mcts::Node), 64U);
    EXPECT_EQ(sizeof(mcts::Edge), 16U);
}

} // namespace
