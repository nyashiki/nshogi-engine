//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../math/score.h"
#include "../mcts/node.h"
#include "../protocol/usilogger.h"

#include <gtest/gtest.h>

#include <random>
#include <thread>
#include <vector>

namespace {

using nshogi::engine::mcts::Node;
namespace score = nshogi::engine::math::score;

} // namespace

TEST(Score, NodeFitsOneCacheLine) {
    EXPECT_EQ(sizeof(Node), 64U);
}

TEST(Score, AveragesWinAndDrawReturnsBeforeApplyingDrawValue) {
    Node N(nullptr);
    N.updateAncestors<false>(1.0f, 0.0f);
    N.updateAncestors<false>(0.5f, 1.0f);

    EXPECT_EQ(N.getVisitsAndVirtualLoss(), 2U);
    EXPECT_DOUBLE_EQ(N.getExpectedScoreAccumulated(), 1.5);
    EXPECT_DOUBLE_EQ(N.getDrawRateAccumulated(), 1.0);
    // One win and one draw average to .75, not the old result .625.
    EXPECT_DOUBLE_EQ(N.getScore(0.5), 0.75);
    EXPECT_DOUBLE_EQ(N.getScore(0.2), 0.6);
    EXPECT_DOUBLE_EQ(N.getScore(0.8), 0.9);
    EXPECT_DOUBLE_EQ(N.getScoreFromParent(0.5, 2, 2), 0.25);
    EXPECT_DOUBLE_EQ(N.getScoreFromParent(0.8, 2, 2), 0.4);
}

TEST(Score, BackupFlipsScoreButPreservesDrawAndRawPredictions) {
    Node Root(nullptr);
    Node Child(&Root);
    Node Grandchild(&Child);
    Node Leaf(&Grandchild);
    Leaf.setEvaluation(nullptr, 0.75f, 0.5f);

    for (int I = 0; I < 3; ++I) {
        Leaf.updateAncestors<false>(Leaf.getWinRatePredicted(),
                                    Leaf.getDrawRatePredicted());
    }

    EXPECT_DOUBLE_EQ(Leaf.getScore(0.5), 0.625);
    EXPECT_DOUBLE_EQ(Grandchild.getScore(0.5), 0.375);
    EXPECT_DOUBLE_EQ(Child.getScore(0.5), 0.625);
    EXPECT_DOUBLE_EQ(Root.getScore(0.5), 0.375);
    for (const Node* N : {&Root, &Child, &Grandchild, &Leaf}) {
        EXPECT_EQ(N->getVisitsAndVirtualLoss(), 3U);
        EXPECT_DOUBLE_EQ(N->getDrawRateAccumulated(), 1.5);
    }
    EXPECT_FLOAT_EQ(Leaf.getWinRatePredicted(), 0.75f);
    EXPECT_FLOAT_EQ(Leaf.getDrawRatePredicted(), 0.5f);
}

TEST(Score, CertainDrawIgnoresConditionalWinRate) {
    for (float ConditionalWinRate : {0.0f, 0.5f, 1.0f}) {
        Node Parent(nullptr);
        Node Child(&Parent);
        Child.updateAncestors<false>(ConditionalWinRate, 1.0f);
        EXPECT_DOUBLE_EQ(Child.getExpectedScoreAccumulated(), 0.5);
        EXPECT_DOUBLE_EQ(Parent.getExpectedScoreAccumulated(), 0.5);

        for (double DrawValue : {0.0, 0.2, 0.5, 0.8, 1.0}) {
            EXPECT_DOUBLE_EQ(Child.getScore(DrawValue), DrawValue);
            EXPECT_DOUBLE_EQ(Child.getScoreFromParent(DrawValue, 1, 1),
                             DrawValue);
        }
    }
}

TEST(Score, WithoutDrawsMatchesOrdinaryWinRate) {
    Node N(nullptr);
    N.updateAncestors<false>(0.25f, 0.0f);
    for (double DrawValue : {0.0, 0.2, 0.5, 0.8, 1.0}) {
        EXPECT_DOUBLE_EQ(N.getScore(DrawValue), 0.25);
        EXPECT_DOUBLE_EQ(N.getScoreFromParent(DrawValue, 1, 1), 0.75);
    }
}

TEST(Score, VirtualLossPenalizesTheEntireParentScore) {
    Node Parent(nullptr);
    Node Child(&Parent);
    Child.updateAncestors<false>(0.0f, 0.0f);
    Child.updateAncestors<false>(0.5f, 1.0f);
    // Parent has one win and one draw worth .8: total 1.8 points.
    EXPECT_DOUBLE_EQ(Child.getScoreFromParent(0.8, 2, 2), 0.9);
    for (int I = 0; I < 2; ++I) {
        Child.incrementVirtualLoss();
        Parent.incrementVirtualLoss();
    }
    EXPECT_DOUBLE_EQ(Child.getScoreFromParent(0.8, 2, 4), 0.45);
    // Selfplay/root statistics exclude virtual visits.
    EXPECT_DOUBLE_EQ(Parent.getScore(0.8), 0.9);

    Child.updateAncestors(0.5f, 1.0f);
    for (const Node* N : {&Parent, &Child}) {
        EXPECT_EQ(N->getVisitsAndVirtualLoss() & Node::VisitMask, 3U);
        EXPECT_EQ(N->getVisitsAndVirtualLoss() >> Node::VirtualLossShift, 1U);
        EXPECT_DOUBLE_EQ(N->getDrawRateAccumulated(), 2.0);
    }
    EXPECT_DOUBLE_EQ(Child.getScoreFromParent(0.8, 3, 4), 0.65);

    Node Draw(nullptr);
    Draw.updateAncestors<false>(0.5f, 1.0f);
    EXPECT_DOUBLE_EQ(Draw.getScoreFromParent(0.8, 1, 2), 0.4);
}

TEST(Score, MixedPredictionsAgreeWithAverageOutcomeProbabilities) {
    std::mt19937 Rng(5090);
    std::uniform_real_distribution<float> Distribution(0.0f, 1.0f);
    for (int Trial = 0; Trial < 100; ++Trial) {
        Node Parent(nullptr);
        Node Child(&Parent);
        double WinSum = 0.0;
        double LossSum = 0.0;
        double DrawSum = 0.0;
        constexpr uint64_t Visits = 32;
        for (uint64_t I = 0; I < Visits; ++I) {
            const float C = Distribution(Rng);
            const float D = Distribution(Rng);
            Child.updateAncestors<false>(C, D);
            WinSum += (1.0 - (double)D) * (double)C;
            LossSum += (1.0 - (double)D) * (1.0 - (double)C);
            DrawSum += (double)D;
        }
        for (double DrawValue : {0.0, 0.2, 0.5, 0.8, 1.0}) {
            EXPECT_NEAR(Child.getScore(DrawValue),
                        (WinSum + DrawValue * DrawSum) / (double)Visits, 1e-12);
            EXPECT_NEAR(Parent.getScore(DrawValue),
                        (LossSum + DrawValue * DrawSum) / (double)Visits,
                        1e-12);
            EXPECT_NEAR(Child.getScoreFromParent(DrawValue, Visits, Visits),
                        Parent.getScore(DrawValue), 1e-12);
        }
    }
}

TEST(Score, ConcurrentBackupsKeepScoreAndVisitTotals) {
    Node Parent(nullptr);
    Node Leaf(&Parent);
    std::vector<std::thread> Threads;
    for (int T = 0; T < 4; ++T) {
        Threads.emplace_back([&Leaf, T]() {
            for (int I = 0; I < 1000; ++I) {
                Leaf.updateAncestors<false>(T % 2 == 0 ? 1.0f : 0.5f,
                                            T % 2 == 0 ? 0.0f : 1.0f);
            }
        });
    }
    for (auto& Thread : Threads) {
        Thread.join();
    }
    EXPECT_EQ(Leaf.getVisitsAndVirtualLoss(), 4000U);
    EXPECT_EQ(Parent.getVisitsAndVirtualLoss(), 4000U);
    EXPECT_DOUBLE_EQ(Leaf.getExpectedScoreAccumulated(), 3000.0);
    EXPECT_DOUBLE_EQ(Parent.getExpectedScoreAccumulated(), 1000.0);
    EXPECT_DOUBLE_EQ(Leaf.getDrawRateAccumulated(), 2000.0);
    EXPECT_DOUBLE_EQ(Parent.getDrawRateAccumulated(), 2000.0);
}

TEST(Score, FallbackRecoversConditionalRateFromSearchStatistics) {
    // These are the aggregates after one win/loss and one draw. Using E
    // directly as the conditional rate would apply the draw adjustment twice.
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(0.75, 0.5), 1.0);
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(0.25, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(0.5, 1.0), 0.5);
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(0.5, 0.0), 0.5);

    const double D = 1.0 - 1.0 / 1024;
    EXPECT_DOUBLE_EQ(
        score::toConditionalWinRate(score::toExpectedScore(0.25, D), D), 0.25);
    // Independently read atomic counters may transiently be inconsistent.
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(0.0, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(score::toConditionalWinRate(1.0, 0.5), 1.0);
}
