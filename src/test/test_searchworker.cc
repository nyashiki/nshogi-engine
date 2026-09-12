//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../allocator/default.h"
#include "../contextmanager.h"
#include "../mcts/searchworker.h"
#include "../mcts/tree.h"

#include <gtest/gtest.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace nshogi;
using namespace nshogi::engine;

class FailOnceAllocator : public allocator::DefaultAllocator {
 public:
    std::function<void()> BeforeFailure;

    void* malloc(std::size_t Size) override {
        if (BeforeFailure) {
            // The callback can perform another descent before this one fails.
            auto Callback = std::exchange(BeforeFailure, {});
            Callback();
            return nullptr;
        }
        return DefaultAllocator::malloc(Size);
    }
};

class SelectionProbe : public mcts::SearchWorker {
 public:
    SelectionProbe(allocator::Allocator* Allocator, mcts::Statistics* Stat)
        : SearchWorker(false, Allocator, Allocator, nullptr, nullptr, Stat) {
    }

    using SearchWorker::collectOneLeaf;
    using SearchWorker::computeUCBMaxEdge;

    const core::State& state() const {
        return *State;
    }

    mcts::Edge* select(mcts::Node* N, bool RegardUnvisitedWin = false) {
        const uint64_t Previous = N->incrementVirtualLoss();
        auto* E = computeUCBMaxEdge(
            N, N->getNumChildren(), Previous >> mcts::Node::VirtualLossShift,
            RegardUnvisitedWin);
        N->decrementVirtualLoss();
        return E;
    }
};

class UCBSelection : public testing::Test {
 protected:
    FailOnceAllocator Allocator;
    mcts::GarbageCollector GC{1, &Allocator, &Allocator};
    mcts::Tree Tree{&GC, &Allocator, nullptr};
    mcts::Statistics Stat;
    SelectionProbe Worker{&Allocator, &Stat};
    core::State Initial = core::StateBuilder::getInitialState();
    core::StateConfig Config;

    mcts::Node* makeRoot() {
        Config.BlackDrawValue = Config.WhiteDrawValue = 0.5f;
        auto* Root = Tree.updateRoot(Initial, false);
        Root->expand(core::MoveGenerator::generateLegalMoves(Initial),
                     &Allocator);
        for (uint16_t I = 0; I < Root->getNumChildren(); ++I) {
            Root->getEdge()[I].setProbability(
                I == 0 ? 0.4f : (I < 3 ? 0.3f : 0.0f));
        }
        Root->setEvaluation(nullptr, 0.5f, 0.0f);
        Root->updateAncestors<false>(0.5f, 0.0f);
        Worker.updateRoot(Initial, Config, Root);
        return Root;
    }

    void finishLeaf(mcts::Node* Leaf, const core::State& State) {
        ASSERT_GT(Leaf->expand(core::MoveGenerator::generateLegalMoves(State),
                               &Allocator), 0);
        std::vector<float> Policy(Leaf->getNumChildren(),
                                  1.0f / Leaf->getNumChildren());
        Leaf->setEvaluation(Policy.data(), 0.1f, 0.0f);
        Leaf->sort();
        Leaf->updateAncestors(0.1f, 0.0f);
    }
};

class PausedUCBSelection : public UCBSelection,
                           public testing::WithParamInterface<bool> {};

TEST_F(UCBSelection, DescentAcquiresConcurrentRootEvaluation) {
    auto* Root = Tree.updateRoot(Initial, false);
    Root->incrementVirtualLoss();
    Worker.updateRoot(Initial, Config, Root);
    const auto Moves = core::MoveGenerator::generateLegalMoves(Initial);
    const std::vector<float> Policy(Moves.size(), 1.0f / (float)Moves.size());
    std::atomic<bool> Start{false};
    mcts::Node* Leaf = nullptr;
    std::thread Writer([&]() {
        while (!Start.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }
        Root->expand(Moves, &Allocator);
        Root->setEvaluation(Policy.data(), 0.5f, 0.0f);
        Root->sort();
        Root->updateAncestors(0.5f, 0.0f);
    });
    std::thread Reader([&]() {
        // No test mutex/acquire may hide collectOneLeaf's publication read.
        Start.store(true, std::memory_order_relaxed);
        while ((Leaf = Worker.collectOneLeaf()) == nullptr) {
            std::this_thread::yield();
        }
    });
    Writer.join();
    Reader.join();
    ASSERT_NE(Leaf, nullptr);
    EXPECT_NE(Leaf, Root);
    EXPECT_EQ(Leaf->getParent(), Root);
    EXPECT_EQ(Leaf, Root->getEdge()[0].getTarget());
    Leaf->decrementVirtualLoss();
    Root->decrementVirtualLoss();
}

class PVRecordingLogger : public logger::Logger {
 public:
    mutable std::vector<logger::PVLog> Logs;

    void printPVLog(const logger::PVLog& Log) const override {
        Logs.push_back(Log);
    }
    void printBestMove(core::Move32) const override {}
    void printLog(const char*) const override {}
    void printStatistics(const mcts::Statistics&) const override {}
    void setIsInverse(bool) override {}
};

struct StopGate {
    std::mutex Mutex;
    std::condition_variable CV;
    bool Entered = false;
    bool Released = false;

    void block() {
        std::unique_lock<std::mutex> Lock(Mutex);
        Entered = true;
        CV.notify_all();
        EXPECT_TRUE(CV.wait_for(Lock, std::chrono::seconds(5),
                                [&]() { return Released; }));
    }

    bool awaitEntry() {
        std::unique_lock<std::mutex> Lock(Mutex);
        return CV.wait_for(Lock, std::chrono::seconds(5),
                            [&]() { return Entered; });
    }

    void release() {
        std::lock_guard<std::mutex> Lock(Mutex);
        Released = true;
        CV.notify_all();
    }
};

class PausingFinalPVLogger : public PVRecordingLogger {
 public:
    mutable StopGate Gate;
    mutable bool First = true;

    void printPVLog(const logger::PVLog& Log) const override {
        if (First) {
            First = false;
            Gate.block();
        }
        PVRecordingLogger::printPVLog(Log);
    }
};

TEST_F(UCBSelection, StopIsIssuedOnlyOncePerSearch) {
    auto* Root = makeRoot();
    ContextManager Context;
    Context.setMaximumThinkinTimeMilliSeconds(0);
    auto Logger = std::make_shared<PausingFinalPVLogger>();
    std::atomic<int> Calls{0};
    std::promise<void> FirstCallback, SecondCallback;
    auto FirstDone = FirstCallback.get_future();
    auto SecondDone = SecondCallback.get_future();
    mcts::SearchWorkerMaster* MasterPtr = nullptr;
    mcts::SearchWorkerMaster Master(
        Context.getContext(), false, &Allocator, &Allocator, nullptr, nullptr,
        &Stat,
        [&]() {
            MasterPtr->stop();
            const int Count = ++Calls;
            if (Count == 1) FirstCallback.set_value();
            if (Count == 2) SecondCallback.set_value();
        },
        Logger);
    MasterPtr = &Master;
    Master.updateRoot(Initial, Config, Root);
    Master.setLimit(NoLimit);
    Master.start();
    ASSERT_TRUE(Logger->Gate.awaitEntry());
    ASSERT_EQ(FirstDone.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);

    // Automatic stop has already run, but the master is still printing its
    // last PV. External stops must not enqueue another callback for this run.
    ASSERT_TRUE(Master.isRunning());
    for (int I = 0; I < 16; ++I) {
        Master.issueStop();
    }
    EXPECT_EQ(SecondDone.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    Logger->Gate.release();
    Master.await();
    EXPECT_EQ(Calls.load(), 1);

    // A new search must still accept its own stop notification.
    Master.start();
    Master.await();
    EXPECT_EQ(Calls.load(), 2);
}

TEST_F(UCBSelection, AwaitIncludesStopCallbackAfterLoopBecomesIdle) {
    auto* Root = makeRoot();
    ContextManager Context;
    Context.setMaximumThinkinTimeMilliSeconds(0);
    auto Logger = std::make_shared<PVRecordingLogger>();
    StopGate CallbackGate;
    std::atomic<bool> CallbackFinished{false};
    mcts::SearchWorkerMaster* MasterPtr = nullptr;
    mcts::SearchWorkerMaster Master(
        Context.getContext(), false, &Allocator, &Allocator, nullptr, nullptr,
        &Stat,
        [&]() {
            MasterPtr->stop();
            CallbackGate.block();
            CallbackFinished.store(true);
        },
        Logger);
    MasterPtr = &Master;
    Master.updateRoot(Initial, Config, Root);
    Master.setLimit(NoLimit);
    Master.start();
    ASSERT_TRUE(CallbackGate.awaitEntry());
    // Deliberately await only the loop to establish that Idle alone is not
    // sufficient. The normal virtual await must still wait for the callback.
    Master.SearchWorker::await();
    worker::Worker& Base = Master;
    auto Done = std::async(std::launch::async, [&]() { Base.await(); });
    EXPECT_EQ(Done.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    EXPECT_FALSE(CallbackFinished.load());
    CallbackGate.release();
    Done.get();
    EXPECT_TRUE(CallbackFinished.load());
}

class PVPublication : public UCBSelection,
                      public testing::WithParamInterface<int> {};

TEST_P(PVPublication, ReadsOnlyEvaluatedNodes) {
    // 0: pending root; 1: evaluated root and pending child; 2: both evaluated.
    const int EvaluatedDepth = GetParam();
    auto* Root = Tree.updateRoot(Initial, false);
    Root->incrementVirtualLoss();
    const auto Moves = core::MoveGenerator::generateLegalMoves(Initial);
    std::vector<float> Policy(Moves.size(), 1.0f / (float)Moves.size());
    Root->expand(Moves, &Allocator);
    Root->setEvaluation(Policy.data(), 0.5f, 0.0f);
    mcts::Node* Pending = Root;
    if (EvaluatedDepth > 0) {
        Root->updateAncestors(0.5f, 0.0f);
        auto NextState = Initial.clone();
        NextState.doMove(Moves[0]);
        mcts::Pointer<mcts::Node> Child;
        Child.malloc(&Allocator, Root);
        Pending = Child.get();
        Root->incrementVirtualLoss();
        Pending->incrementVirtualLoss();
        Root->publishChild(&Root->getEdge()[0], std::move(Child));
        const auto ChildMoves = core::MoveGenerator::generateLegalMoves(NextState);
        Policy.assign(ChildMoves.size(), 1.0f / (float)ChildMoves.size());
        Pending->expand(ChildMoves, &Allocator);
        Pending->setEvaluation(Policy.data(), 0.5f, 0.0f);
        if (EvaluatedDepth == 2) {
            Pending->updateAncestors(0.5f, 0.0f);
        }
    }

    ContextManager Context;
    // Stop before a descent, but exercise the real master's final PV log.
    Context.setMinimumThinkinTimeMilliSeconds(0);
    Context.setMaximumThinkinTimeMilliSeconds(0);
    auto Logger = std::make_shared<PVRecordingLogger>();
    mcts::SearchWorkerMaster* MasterPtr = nullptr;
    {
        mcts::SearchWorkerMaster Master(
            Context.getContext(), false, &Allocator, &Allocator, nullptr,
            nullptr, &Stat, [&]() { MasterPtr->stop(); }, Logger);
        MasterPtr = &Master;
        Master.updateRoot(Initial, Config, Root);
        Master.setLimit(NoLimit);
        Master.start();
        Master.await();
    }
    ASSERT_FALSE(Logger->Logs.empty());
    for (const auto& Log : Logger->Logs) {
        EXPECT_EQ(Log.PV.size(), (std::size_t)EvaluatedDepth);
    }
    if (EvaluatedDepth < 2) {
        Pending->decrementVirtualLoss();
        if (Pending != Root) {
            Root->decrementVirtualLoss();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(PendingEvaluation, PVPublication,
                         testing::Values(0, 1, 2));

TEST_F(UCBSelection, SlowRootEvaluationRemainsPrivateToItsOwner) {
    auto* Root = Tree.updateRoot(Initial, false);
    Root->incrementVirtualLoss();
    const auto Moves = core::MoveGenerator::generateLegalMoves(Initial);
    const std::vector<float> Policy(Moves.size(), 1.0f / (float)Moves.size());
    ContextManager Context;
    Context.setMinimumThinkinTimeMilliSeconds(0);
    Context.setMaximumThinkinTimeMilliSeconds(1000);
    auto Logger = std::make_shared<PVRecordingLogger>();
    mcts::SearchWorkerMaster* MasterPtr = nullptr;
    {
        mcts::SearchWorkerMaster Master(
            Context.getContext(), false, &Allocator, &Allocator, nullptr,
            nullptr, &Stat, [&]() { MasterPtr->stop(); }, Logger);
        MasterPtr = &Master;
        Master.updateRoot(Initial, Config, Root);
        Limit Lim;
        Lim.ByoyomiMilliSeconds = 2000;
        Master.setLimit(Lim);
        Master.start();
        std::thread Writer([&]() {
            // Cross hasMadeUpMind's 470 ms threshold before expansion. Keep
            // the first evaluation pending throughout this timed search.
            std::this_thread::sleep_for(std::chrono::milliseconds(550));
            Root->expand(Moves, &Allocator);
            Root->setEvaluation(Policy.data(), 0.5f, 0.0f);
            Root->sort();
        });
        Master.await();
        Writer.join();
    }
    ASSERT_FALSE(Logger->Logs.empty());
    for (const auto& Log : Logger->Logs) {
        EXPECT_TRUE(Log.PV.empty());
        EXPECT_EQ(Log.NumNodes, 0U);
    }
    EXPECT_EQ(Root->getVisitsAndVirtualLoss(), 1ULL << mcts::Node::VirtualLossShift);
    Root->decrementVirtualLoss();
}

TEST_P(PausedUCBSelection, LaterEvaluatedChildIsConsidered) {
    auto* Root = makeRoot();
    const uint64_t BeforeA = Root->incrementVirtualLoss();
    auto* EdgeA = Worker.computeUCBMaxEdge(
        Root, Root->getNumChildren(),
        BeforeA >> mcts::Node::VirtualLossShift, false);
    ASSERT_EQ(EdgeA, &Root->getEdge()[0]);
    // Pause descent A on either side of markExpanding(). Descent B can
    // complete in both schedules while A's virtual loss remains in place.
    if (GetParam()) {
        EXPECT_FALSE(EdgeA->markExpanding());
    }

    SelectionProbe B(&Allocator, &Stat);
    B.updateRoot(Initial, Config, Root);
    auto* LeafB = B.collectOneLeaf();
    ASSERT_NE(LeafB, nullptr);
    ASSERT_EQ(LeafB, Root->getEdge()[1].getTarget());
    finishLeaf(LeafB, B.state());
    EXPECT_EQ(Root->getVisitsAndVirtualLoss(),
              2ULL | (1ULL << mcts::Node::VirtualLossShift));
    EXPECT_EQ(Worker.select(Root), &Root->getEdge()[1]);

    Root->decrementVirtualLoss();
    if (GetParam()) {
        EdgeA->unmarkExpanding();
    }
    EXPECT_EQ(Root->getVisitsAndVirtualLoss(), 2U);
}

INSTANTIATE_TEST_SUITE_P(
    ExpansionBoundary, PausedUCBSelection, testing::Bool(),
    [](const testing::TestParamInfo<bool>& Info) {
        return Info.param ? "AfterClaim" : "BeforeClaim";
    });

TEST_F(UCBSelection, AllocationFailureDoesNotHideLaterEvaluatedChild) {
    auto* Root = makeRoot();
    SelectionProbe B(&Allocator, &Stat);
    B.updateRoot(Initial, Config, Root);

    // A has marked edge 0 when its allocator runs this callback. Let B
    // publish and back up edge 1, then take the real allocation-failure path.
    Allocator.BeforeFailure = [&] {
        EXPECT_TRUE(Root->getEdge()[0].isExpanding());
        auto* LeafB = B.collectOneLeaf();
        ASSERT_NE(LeafB, nullptr);
        ASSERT_EQ(LeafB, Root->getEdge()[1].getTarget());
        finishLeaf(LeafB, B.state());
    };
    EXPECT_EQ(Worker.collectOneLeaf(), nullptr);
    EXPECT_EQ(Root->getEdge()[0].getTarget(), nullptr);
    EXPECT_FALSE(Root->getEdge()[0].isExpanding());
    EXPECT_EQ(Root->getVisitsAndVirtualLoss(), 2U);
    EXPECT_EQ(Root->getExpandedEnd(), 2);

    Worker.updateRoot(Initial, Config, Root);
    EXPECT_EQ(Worker.select(Root), &Root->getEdge()[1]);
}

TEST_F(UCBSelection, ProvenWinAfterAnUnvisitedMoveIsConsidered) {
    auto* Root = makeRoot();
    mcts::Pointer<mcts::Node> Child;
    auto* N = Child.malloc(&Allocator, Root);
    N->setPlyToTerminalSolved(-3);
    Root->publishChild(&Root->getEdge()[2], std::move(Child));
    N->updateAncestors<false>(0.0f, 0.0f);
    EXPECT_EQ(Worker.select(Root), &Root->getEdge()[2]);
    EXPECT_EQ(Root->getPlyToTerminalSolved(), 4);
}

TEST_F(UCBSelection, PendingChildAfterAnUnvisitedMoveIsSkipped) {
    auto* Root = makeRoot();
    mcts::Pointer<mcts::Node> Child;
    auto* N = Child.malloc(&Allocator, Root);
    Root->publishChild(&Root->getEdge()[1], std::move(Child));
    N->updateAncestors<false>(0.1f, 0.0f);

    mcts::Pointer<mcts::Node> Pending;
    auto* P = Pending.malloc(&Allocator, Root);
    P->incrementVirtualLoss();
    Root->incrementVirtualLoss();
    Root->publishChild(&Root->getEdge()[2], std::move(Pending));
    EXPECT_EQ(Worker.select(Root), &Root->getEdge()[1]);
    P->decrementVirtualLoss();
    Root->decrementVirtualLoss();
}

TEST_F(UCBSelection, EveryExpansionPatternMatchesFullScan) {
    constexpr uint16_t Count = 8;
    // Exhaust all nonempty subsets, including holes before and between
    // evaluated children. Compare against PUCT over every move, with no
    // early-exit assumption and with virtual losses on evaluated children.
    for (uint16_t Mask = 1; Mask < (1U << Count); ++Mask) {
        SCOPED_TRACE(Mask);
        auto* Root = makeRoot();
        std::array<uint16_t, Count> Visits{};
        std::array<uint16_t, Count> VirtualLosses{};
        std::array<double, Count> ParentScores{};
        uint64_t TotalVisits = 1;
        uint64_t TotalVirtualLosses = 0;
        for (uint16_t I = 0; I < Root->getNumChildren(); ++I) {
            Root->getEdge()[I].setProbability(
                I < Count ? static_cast<float>(Count - I) / 36.0f : 0.0f);
        }
        // Publish in reverse order to exercise decreasing indices too.
        for (int I = Count - 1; I >= 0; --I) {
            const auto Index = static_cast<uint16_t>(I);
            if ((Mask & (1U << Index)) == 0) {
                continue;
            }
            Visits[Index] = static_cast<uint16_t>(1 + Index % 5);
            VirtualLosses[Index] = static_cast<uint16_t>(Index % 3);
            // A strong, low-prior child makes premature scan termination
            // observably wrong (e.g. Mask == 128), even with score tolerance.
            const float OwnWin = Index == Count - 1
                                     ? 0.0f
                                     : static_cast<float>((Index * 7) % 11) / 10.0f;
            ParentScores[Index] = 1.0 - static_cast<double>(OwnWin);
            mcts::Pointer<mcts::Node> Child;
            auto* N = Child.malloc(&Allocator, Root);
            Root->publishChild(&Root->getEdge()[Index], std::move(Child));
            for (uint16_t V = 0; V < Visits[Index]; ++V) {
                N->updateAncestors<false>(OwnWin, 0.0f);
            }
            for (uint16_t V = 0; V < VirtualLosses[Index]; ++V) {
                N->incrementVirtualLoss();
                Root->incrementVirtualLoss();
            }
            TotalVisits += Visits[Index];
            TotalVirtualLosses += VirtualLosses[Index];
        }

        const double ParentN =
            static_cast<double>(TotalVisits + TotalVirtualLosses);
        const double C =
            (std::log((ParentN + 19652.0) / 19652.0) + 1.25) *
            std::sqrt(ParentN);
        for (bool Optimistic : {false, true}) {
            SCOPED_TRACE(Optimistic);
            std::vector<double> Values(Root->getNumChildren());
            double Best = std::numeric_limits<double>::lowest();
            uint16_t BestIndex = 0;
            for (uint16_t I = 0; I < Root->getNumChildren(); ++I) {
                const double P = Root->getEdge()[I].getProbability();
                double Value = (Optimistic ? 1.0 : 0.0) + C * P;
                if (I < Count && Visits[I] != 0) {
                    const double N = Visits[I] + VirtualLosses[I];
                    Value = ParentScores[I] * Visits[I] / N +
                            C * P / (1.0 + N);
                }
                Values[I] = Value;
                if (Value > Best) {
                    Best = Value;
                    BestIndex = I;
                }
            }
            const auto* Selected = Worker.select(Root, Optimistic);
            ASSERT_NE(Selected, nullptr);
            const auto SelectedIndex = static_cast<std::size_t>(
                Selected - Root->getEdge().get());
            ASSERT_LT(SelectedIndex, Values.size());
            // Equivalent maxima can round differently with FMA or
            // reassociation. Mask 79, Optimistic=true ties edges 0 and 4:
            // 1 + C * P[0] / 2 == 1 + C * P[4]. Compare the selected
            // score, allowing only double-precision rounding error.
            const double Tolerance =
                16.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(Best));
            EXPECT_NEAR(Values[SelectedIndex], Best, Tolerance)
                << "selected=" << SelectedIndex << " best=" << BestIndex;
        }

        for (uint16_t I = 0; I < Count; ++I) {
            for (uint16_t V = 0; V < VirtualLosses[I]; ++V) {
                Root->getEdge()[I].getTarget()->decrementVirtualLoss();
                Root->decrementVirtualLoss();
            }
        }
    }
}

} // namespace
