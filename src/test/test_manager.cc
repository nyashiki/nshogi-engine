//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../contextmanager.h"
#include "../mcts/manager.h"
#include "../protocol/usilogger.h"

#include <gtest/gtest.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/sfen.h>
#include <nshogi/solver/dfs.h>

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>

// These integration tests use a CPU evaluator and need no model or GPU.
#if defined(EXECUTOR_RANDOM) || defined(EXECUTOR_ZERO)

namespace {

using namespace nshogi;
using namespace nshogi::engine;
using namespace std::chrono_literals;

class SilentLogger : public logger::Logger {
 public:
    void printPVLog(const logger::PVLog&) const override {
    }
    void printBestMove(core::Move32) const override {
    }
    void printLog(const char*) const override {
    }
    void printStatistics(const mcts::Statistics&) const override {
    }
    void setIsInverse(bool) override {
    }
};

struct FinishGate {
    std::mutex Mutex;
    std::condition_variable CV;
    bool Entered = false;
    bool Released = false;

    void block() {
        std::unique_lock<std::mutex> Lock(Mutex);
        Entered = true;
        CV.notify_all();
        EXPECT_TRUE(CV.wait_for(Lock, 5s, [&]() { return Released; }));
    }

    bool awaitEntry() {
        std::unique_lock<std::mutex> Lock(Mutex);
        return CV.wait_for(Lock, 5s, [&]() { return Entered; });
    }

    void release() {
        std::lock_guard<std::mutex> Lock(Mutex);
        Released = true;
        CV.notify_all();
    }
};

struct Replies {
    std::mutex Mutex;
    std::condition_variable CV;
    std::vector<std::pair<int, core::Move32>> Moves;

    void add(int Request, core::Move32 Move) {
        std::lock_guard<std::mutex> Lock(Mutex);
        Moves.emplace_back(Request, Move);
        CV.notify_all();
    }

    bool awaitSize(std::size_t Size) {
        std::unique_lock<std::mutex> Lock(Mutex);
        return CV.wait_for(Lock, 5s, [&]() { return Moves.size() >= Size; });
    }
};

class PausingPonderLogger : public protocol::usi::USILogger {
 public:
    std::shared_ptr<FinishGate> Gate = std::make_shared<FinishGate>();
    std::atomic<bool> Pondering{false};
    mutable std::atomic<bool> Paused{false};

    void setIsInverse(bool Value) override {
        USILogger::setIsInverse(Value);
        Pondering.store(Value, std::memory_order_relaxed);
    }

    void printPVLog(const logger::PVLog& Log) const override {
        if (Pondering.load(std::memory_order_relaxed) &&
            !Paused.exchange(true, std::memory_order_relaxed)) {
            // Keep the actual master running after the supervisor has
            // finished, so queuing the next go cannot rely on its mutex
            // alone to protect the master's Limit or log viewpoint.
            Gate->block();
        }
        USILogger::printPVLog(Log);
    }
};

void configure(ContextManager& C, bool Ponder) {
    C.setAvailableMemoryMB(32);
    C.setEvalCacheMemoryMB(1);
    C.setNumSearchThreads(2);
    C.setNumEvaluationThreadsPerGPU(1);
    C.setNumFeedThreads(1);
    C.setNumCheckmateSearchThreads(0);
    C.setNumGarbageCollectorThreads(1);
    C.setBatchSize(4);
    C.setIsPonderingEnabled(Ponder);
    C.setBookEnabled(false);
    C.setPrintStatistics(false);
    C.setThinkingTimeMargin(0);
    C.setMinimumThinkinTimeMilliSeconds(0);
    C.setMaximumThinkinTimeMilliSeconds(1000);
}

bool isLegal(const core::State& State, core::Move32 Move) {
    for (auto LegalMove : core::MoveGenerator::generateLegalMoves(State)) {
        if (Move == LegalMove) {
            return true;
        }
    }
    return false;
}

class ManagerSwitch : public testing::TestWithParam<bool> {};

TEST_P(ManagerSwitch, PreviousCallbacksSurviveReplacementDuringFinalization) {
    ContextManager C;
    configure(C, GetParam());
    const auto State = core::StateBuilder::getInitialState();
    core::StateConfig Config;
    Limit Lim;
    Lim.ByoyomiMilliSeconds = 10;
    auto Gate = std::make_shared<FinishGate>();
    Replies Completed;

    // Destroy the manager before the callback state it references.
    {
        mcts::Manager Manager(C.getContext(), std::make_shared<SilentLogger>());
        Manager.thinkNextMove(
            State, Config, Lim,
            [&](core::Move32 Move) { Completed.add(0, Move); },
            [Gate](mcts::Tree*) {
                // Keep the gate alive even in the buggy implementation,
                // which replaces this callback while it is executing.
                auto HeldGate = Gate;
                HeldGate->block();
            });
        ASSERT_TRUE(Gate->awaitEntry());

        Manager.interrupt();
        std::promise<void> Started;
        auto Replacement = std::async(std::launch::async, [&]() {
            Started.set_value();
            Manager.thinkNextMove(State, Config, Lim, [&](core::Move32 Move) {
                Completed.add(1, Move);
            });
        });
        Started.get_future().wait();
        // The next go must wait until the old supervisor task has finished.
        EXPECT_EQ(Replacement.wait_for(100ms), std::future_status::timeout);
        Gate->release();
        Replacement.get();
        EXPECT_TRUE(Completed.awaitSize(2));
    }

    ASSERT_EQ(Completed.Moves.size(), 2U);
    EXPECT_EQ(Completed.Moves[0].first, 0);
    EXPECT_EQ(Completed.Moves[1].first, 1);
    for (const auto& Reply : Completed.Moves) {
        EXPECT_TRUE(isLegal(State, Reply.second));
    }
}

TEST_P(ManagerSwitch, StopThenGoKeepsRepliesWithTheirPositions) {
    ContextManager C;
    configure(C, GetParam());
    auto Black = core::StateBuilder::getInitialState();
    auto White = Black.clone();
    White.doMove(core::MoveGenerator::generateLegalMoves(White)[0]);
    core::StateConfig Config;
    Limit Lim;
    Lim.ByoyomiMilliSeconds = 20;
    Replies Completed;
    constexpr int Requests = 16;

    {
        mcts::Manager Manager(C.getContext(), std::make_shared<SilentLogger>());
        for (int I = 0; I < Requests; ++I) {
            // Alternate unrelated roots and interrupt without waiting for
            // bestmove before submitting the next request.
            const auto& State = I % 2 == 0 ? Black : White;
            Manager.thinkNextMove(
                State, Config, Lim,
                [&, I](core::Move32 Move) { Completed.add(I, Move); });
            Manager.interrupt();
        }
        EXPECT_TRUE(Completed.awaitSize(Requests));
    }

    ASSERT_EQ(Completed.Moves.size(), (std::size_t)Requests);
    for (int I = 0; I < Requests; ++I) {
        EXPECT_EQ(Completed.Moves[(std::size_t)I].first, I);
        EXPECT_TRUE(isLegal(I % 2 == 0 ? Black : White,
                            Completed.Moves[(std::size_t)I].second))
            << "request=" << I
            << " move=" << Completed.Moves[(std::size_t)I].second.value()
            << " isNone=" << Completed.Moves[(std::size_t)I].second.isNone();
    }
}

TEST_P(ManagerSwitch, BookStopDoesNotCancelTheFollowingSearch) {
    ContextManager C;
    configure(C, GetParam());
    C.setMaximumThinkinTimeMilliSeconds(10000);
    const auto Initial = core::StateBuilder::getInitialState();
    const auto Moves = core::MoveGenerator::generateLegalMoves(Initial);
    auto OutsideBook = Initial.clone();
    OutsideBook.doMove(Moves[1]);
    struct TemporaryBook {
        std::string Path =
            (std::filesystem::temp_directory_path() /
             ("nshogi-stop-book-" +
              std::to_string(std::chrono::steady_clock::now()
                                 .time_since_epoch().count()) + ".db"))
                .string();
        ~TemporaryBook() { std::remove(Path.c_str()); }
    } Book;
    {
        std::ofstream Stream(Book.Path);
        ASSERT_TRUE(Stream);
        Stream << "sfen "
               << nshogi::io::sfen::positionToSfen(Initial.getPosition())
               << '\n' << nshogi::io::sfen::move32ToSfen(Moves[0])
               << " none 0 1 1\n";
    }
    C.setBookEnabled(true);
    C.setBookPath(Book.Path);
    core::StateConfig Config;
    Limit BookLimit, SearchLimit;
    // A small limit lets automatic stop overlap the direct book stop.
    BookLimit.NumNodes = 1;
    SearchLimit.NumNodes = 1024;
    Replies Completed;
    uint64_t Visits = 0;
    {
        mcts::Manager Manager(C.getContext(), std::make_shared<SilentLogger>());
        for (int I = 0; I < 4; ++I) {
            Manager.thinkNextMove(Initial, Config, BookLimit,
                [&, I](core::Move32 Move) { Completed.add(2 * I, Move); });
            ASSERT_TRUE(Completed.awaitSize((std::size_t)(2 * I + 1)));
            Manager.thinkNextMove(OutsideBook, Config, SearchLimit,
                [&, I](core::Move32 Move) { Completed.add(2 * I + 1, Move); },
                [&](mcts::Tree* Tree) {
                    Visits = Tree->getRoot()->getVisitsAndVirtualLoss() &
                             mcts::Node::VisitMask;
                });
            ASSERT_TRUE(Completed.awaitSize((std::size_t)(2 * I + 2)));
            EXPECT_GE(Visits, SearchLimit.NumNodes);
        }
    }
    ASSERT_EQ(Completed.Moves.size(), 8U);
    for (std::size_t I = 0; I < Completed.Moves.size(); I += 2) {
        EXPECT_EQ(Completed.Moves[I].second, Moves[0]);
        EXPECT_TRUE(isLegal(OutsideBook, Completed.Moves[I + 1].second));
    }
}

INSTANTIATE_TEST_SUITE_P(Pondering, ManagerSwitch, testing::Bool());

TEST(Manager, NextLimitIsAppliedOnlyAfterPonderingStops) {
    ContextManager C;
    configure(C, true);
    // Pause at the master's first pondering iteration. With the default
    // interval, a fast search can hit its memory limit before any PV is
    // logged; pondering suppresses the immediate log on that stop path.
    C.setLogMargin(0);
    C.setMaximumThinkinTimeMilliSeconds(10000);
    const auto Initial = core::StateBuilder::getInitialState();
    core::StateConfig Config;
    auto Logger = std::make_shared<PausingPonderLogger>();
    Replies Completed;
    uint64_t SecondVisits = 0;
    Limit First, Second;
    First.NumNodes = 16;
    Second.NumNodes = 1024;

    {
        mcts::Manager Manager(C.getContext(), Logger);
        Manager.thinkNextMove(Initial, Config, First, [&](core::Move32 Move) {
            Completed.add(0, Move);
        });
        ASSERT_TRUE(Completed.awaitSize(1));
        ASSERT_TRUE(Logger->Gate->awaitEntry());

        // Rewind to the initial root so reused visits cannot satisfy the
        // second request's larger node limit before it starts searching.
        auto Replacement = std::async(std::launch::async, [&]() {
            Manager.thinkNextMove(
                Initial, Config, Second,
                [&](core::Move32 Move) { Completed.add(1, Move); },
                [&](mcts::Tree* Tree) {
                    SecondVisits = Tree->getRoot()->getVisitsAndVirtualLoss() &
                                   mcts::Node::VisitMask;
                });
        });
        // Queuing a request can finish while pondering is still blocked.
        // setLimit() must not be called on that running master (debug assert).
        EXPECT_EQ(Replacement.wait_for(1s), std::future_status::ready);
        // The queued request must not change the old pondering log's view.
        EXPECT_TRUE(Logger->Pondering.load(std::memory_order_relaxed));
        Logger->Gate->release();
        Replacement.get();
        EXPECT_TRUE(Completed.awaitSize(2));
    }

    ASSERT_EQ(Completed.Moves.size(), 2U);
    EXPECT_EQ(Completed.Moves[0].first, 0);
    EXPECT_EQ(Completed.Moves[1].first, 1);
    EXPECT_GE(SecondVisits, Second.NumNodes);
    for (const auto& Reply : Completed.Moves) {
        EXPECT_TRUE(isLegal(Initial, Reply.second));
    }
}

} // namespace

#endif
