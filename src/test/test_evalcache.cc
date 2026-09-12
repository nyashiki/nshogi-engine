//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../mcts/evalcache.h"

#include <atomic>
#include <barrier>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

using nshogi::engine::mcts::EvalCache;

// Every test stores, for a hash H, the entry deterministically derived
// from H below, so any successful load can be checked bit-for-bit
// without keeping a copy of what was stored. A mismatch means the
// cache returned a torn or corrupted entry.
uint16_t numMovesOf(uint64_t H) {
    return (uint16_t)(1 + (H * 0x9E3779B97F4A7C15ULL >> 32) %
                              EvalCache::MAX_CACHE_MOVES_COUNT);
}

float policyOf(uint64_t H, uint32_t I) {
    return (float)(uint32_t)(H * 2654435761U + I * 40503U) / (float)UINT32_MAX;
}

void fillEntry(uint64_t H, EvalCache::EvalInfo* EI) {
    EI->NumMoves = numMovesOf(H);
    for (uint32_t I = 0; I < EI->NumMoves; ++I) {
        EI->Policy[I] = policyOf(H, I);
    }
    EI->WinRate = policyOf(H, 0xFFFF0000U);
    EI->DrawRate = policyOf(H, 0xFFFF0001U);
}

void expectEntryMatches(uint64_t H, const EvalCache::EvalInfo& EI) {
    ASSERT_EQ(EI.NumMoves, numMovesOf(H));
    for (uint32_t I = 0; I < EI.NumMoves; ++I) {
        ASSERT_EQ(EI.Policy[I], policyOf(H, I));
    }
    ASSERT_EQ(EI.WinRate, policyOf(H, 0xFFFF0000U));
    ASSERT_EQ(EI.DrawRate, policyOf(H, 0xFFFF0001U));
}

} // namespace

TEST(EvalCache, StoreLoadRoundTrip) {
    EvalCache Cache(1);

    for (uint16_t NumM : {(uint16_t)1, (uint16_t)2, (uint16_t)43, (uint16_t)128,
                          (uint16_t)EvalCache::MAX_CACHE_MOVES_COUNT}) {
        const uint64_t Hash = 0x123456789ABCDEFULL + NumM;
        float P[EvalCache::MAX_CACHE_MOVES_COUNT];
        for (uint16_t I = 0; I < NumM; ++I) {
            P[I] = (float)I * 0.25f;
        }

        ASSERT_TRUE(Cache.store(Hash, NumM, P, 0.75f, 0.125f));

        EvalCache::EvalInfo EI;
        ASSERT_TRUE(Cache.load(Hash, &EI));
        ASSERT_EQ(EI.NumMoves, NumM);
        ASSERT_EQ(EI.WinRate, 0.75f);
        ASSERT_EQ(EI.DrawRate, 0.125f);
        ASSERT_EQ(std::memcmp(EI.Policy, P, sizeof(float) * NumM), 0);
    }
}

TEST(EvalCache, RejectsOutOfRangeMoveCounts) {
    EvalCache Cache(1);

    float P[EvalCache::MAX_CACHE_MOVES_COUNT] = {};
    ASSERT_FALSE(Cache.store(0x1ULL, 0, P, 0.5f, 0.5f));
    ASSERT_FALSE(Cache.store(0x1ULL,
                             (uint16_t)(EvalCache::MAX_CACHE_MOVES_COUNT + 1),
                             P, 0.5f, 0.5f));

    EvalCache::EvalInfo EI;
    ASSERT_FALSE(Cache.load(0x1ULL, &EI));
}

TEST(EvalCache, MissReturnsFalse) {
    EvalCache Cache(1);

    EvalCache::EvalInfo EI;
    for (uint64_t H = 1; H <= 4096; ++H) {
        ASSERT_FALSE(Cache.load(H, &EI));
    }
}

// Overfill a small cache so that every bucket evicts and compacts many
// times, and check that whatever remains loadable is exactly what was
// stored for that hash. This exercises the variable-length payload
// packing: an eviction or compaction bug shows up as a corrupted
// neighbor entry.
TEST(EvalCache, EvictionAndCompactionKeepEntriesIntact) {
    std::mt19937_64 Rng(20260725);
    EvalCache Cache(1);

    std::vector<uint64_t> Hashes(200000);
    EvalCache::EvalInfo EI;
    for (uint64_t& H : Hashes) {
        H = Rng() | 1U;
        fillEntry(H, &EI);
        ASSERT_TRUE(
            Cache.store(H, EI.NumMoves, EI.Policy, EI.WinRate, EI.DrawRate));

        // Re-loading right after a single-threaded store must hit.
        ASSERT_TRUE(Cache.load(H, &EI));
        expectEntryMatches(H, EI);
    }

    std::size_t NumHits = 0;
    for (const uint64_t H : Hashes) {
        if (Cache.load(H, &EI)) {
            expectEntryMatches(H, EI);
            ++NumHits;
        }
    }
    // The cache is far smaller than the working set, so most entries
    // are gone, but the most recently stored ones must survive.
    ASSERT_GT(NumHits, 0);
}

// Re-storing the same hash must refresh it cheaply and keep the entry
// loadable.
TEST(EvalCache, RestoreSameHash) {
    EvalCache Cache(1);

    const uint64_t H = 0xDEADBEEFCAFEULL;
    EvalCache::EvalInfo EI;
    fillEntry(H, &EI);
    for (int I = 0; I < 3; ++I) {
        ASSERT_TRUE(
            Cache.store(H, EI.NumMoves, EI.Policy, EI.WinRate, EI.DrawRate));
    }
    ASSERT_TRUE(Cache.load(H, &EI));
    expectEntryMatches(H, EI);
}

TEST(EvalCache, ConcurrentReadersKeepHits) {
    EvalCache Cache(1);
    static constexpr uint16_t NumMoves = 43;
    constexpr int NumReaders = 4;
    float Policy[NumMoves];

    // All four entries occupy the same bucket and fit without eviction.
    for (uint64_t H = 1; H <= 4; ++H) {
        for (uint16_t I = 0; I < NumMoves; ++I) {
            Policy[I] = policyOf(H, I);
        }
        ASSERT_TRUE(Cache.store(H, NumMoves, Policy, 0.75f, 0.125f));
    }

    std::barrier Start(NumReaders + 1);
    std::vector<std::thread> Threads;
    for (int T = 0; T < NumReaders; ++T) {
        Threads.emplace_back([&Cache, &Start, T]() {
            EvalCache::EvalInfo EI;
            Start.arrive_and_wait();
            for (uint32_t I = 0; I < 20000; ++I) {
                const uint64_t H = 1 + (I + (uint32_t)T) % 4;
                // Read-only contention must neither evict nor reject a hit.
                ASSERT_TRUE(Cache.load(H, &EI));
                ASSERT_EQ(EI.NumMoves, NumMoves);
                ASSERT_EQ(EI.WinRate, 0.75f);
                ASSERT_EQ(EI.DrawRate, 0.125f);
                for (uint16_t J = 0; J < NumMoves; ++J) {
                    ASSERT_EQ(EI.Policy[J], policyOf(H, J));
                }
            }
        });
    }
    Start.arrive_and_wait();
    for (auto& Thread : Threads) {
        Thread.join();
    }
}

// Concurrent writers and readers on a small hash pool: loads may miss
// under interference, but a load that reports a hit must never return
// torn data.
TEST(EvalCache, ConcurrentIntegrity) {
    EvalCache Cache(4);

    constexpr std::size_t PoolSize = 4096;
    constexpr int NumWriters = 4;
    constexpr int NumReaders = 4;

    std::atomic<bool> Stop(false);
    std::atomic<uint64_t> NumHits(0);
    std::vector<std::thread> Threads;

    for (int T = 0; T < NumWriters; ++T) {
        Threads.emplace_back([&Cache, &Stop, T]() {
            std::mt19937_64 Rng((uint64_t)T + 1);
            EvalCache::EvalInfo EI;
            while (!Stop.load(std::memory_order_relaxed)) {
                const uint64_t H = (Rng() % PoolSize) * 0x100000001ULL + 1;
                fillEntry(H, &EI);
                Cache.store(H, EI.NumMoves, EI.Policy, EI.WinRate, EI.DrawRate);
            }
        });
    }
    for (int T = 0; T < NumReaders; ++T) {
        Threads.emplace_back([&Cache, &Stop, &NumHits, T]() {
            std::mt19937_64 Rng((uint64_t)T + 1001);
            EvalCache::EvalInfo EI;
            while (!Stop.load(std::memory_order_relaxed)) {
                const uint64_t H = (Rng() % PoolSize) * 0x100000001ULL + 1;
                if (Cache.load(H, &EI)) {
                    expectEntryMatches(H, EI);
                    NumHits.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    Stop.store(true);
    for (auto& Th : Threads) {
        Th.join();
    }

    // The pool is small, so readers must actually have hit.
    ASSERT_GT(NumHits.load(), 0ULL);
}
