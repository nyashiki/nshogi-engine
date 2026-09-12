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

#include <algorithm>
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
    return (uint16_t)(1 + (H * 0x9E3779B97F4A7C15ULL >> 32) % 2048);
}

float policyOf(uint64_t H, uint32_t I) {
    return (float)(uint32_t)(H * 2654435761U + I * 40503U) / (float)UINT32_MAX;
}

void fillEntry(uint64_t H, EvalCache::EvalInfo* EI) {
    EI->NumMoves = numMovesOf(H);
    EI->Policy.resize(EI->NumMoves);
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
                          (uint16_t)599, (uint16_t)600, (uint16_t)601,
                          (uint16_t)4096, (uint16_t)UINT16_MAX}) {
        const uint64_t Hash = 0x123456789ABCDEFULL + NumM;
        std::vector<float> P(NumM);
        for (uint16_t I = 0; I < NumM; ++I) {
            P[I] = (float)I * 0.25f;
        }

        ASSERT_TRUE(Cache.store(Hash, NumM, P.data(), 0.75f, 0.125f));

        EvalCache::EvalInfo EI;
        ASSERT_TRUE(Cache.load(Hash, &EI));
        ASSERT_EQ(EI.NumMoves, NumM);
        ASSERT_EQ(EI.WinRate, 0.75f);
        ASSERT_EQ(EI.DrawRate, 0.125f);
        ASSERT_EQ(std::memcmp(EI.Policy.data(), P.data(), sizeof(float) * NumM), 0);
    }
}

TEST(EvalCache, RejectsEmptyPolicy) {
    EvalCache Cache(1);

    float P[] = {0.0f};
    ASSERT_FALSE(Cache.store(0x1ULL, 0, P, 0.5f, 0.5f));

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
            Cache.store(H, EI.NumMoves, EI.Policy.data(), EI.WinRate, EI.DrawRate));

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
    ASSERT_GT(NumHits, std::size_t{0});
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
            Cache.store(H, EI.NumMoves, EI.Policy.data(), EI.WinRate, EI.DrawRate));
    }
    ASSERT_TRUE(Cache.load(H, &EI));
    expectEntryMatches(H, EI);
}

TEST(EvalCache, DuplicateKeepsOriginalPayload) {
    EvalCache Cache(1);
    const float Original[] = {0.25f, -0.0f, 0.75f};
    const float Replacement[] = {0.5f, 0.5f, 0.0f};

    // Zero is a valid hash, although unused ways also start with a zero key.
    ASSERT_TRUE(Cache.store(0, 3, Original, 0.75f, 0.125f));
    ASSERT_TRUE(Cache.store(0, 3, Replacement, 0.25f, 0.5f));

    EvalCache::EvalInfo EI;
    ASSERT_TRUE(Cache.load(0, &EI));
    EXPECT_EQ(EI.NumMoves, 3);
    EXPECT_EQ(EI.WinRate, 0.75f);
    EXPECT_EQ(EI.DrawRate, 0.125f);
    EXPECT_EQ(std::memcmp(EI.Policy.data(), Original, sizeof(Original)), 0);
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

TEST(EvalCache, StorageGrowsAndShrinksWithPolicySizes) {
    EvalCache Cache(1);
    const std::size_t EmptyBytes = Cache.getMemoryUsed();
    const std::vector<float> Large(4096, 0.25f);
    const std::vector<float> Small(43, 0.5f);

    for (uint64_t H = 1; H <= 8; ++H) {
        ASSERT_TRUE(Cache.store(H, (uint16_t)Large.size(), Large.data(),
                                0.75f, 0.125f));
    }
    const std::size_t LargeBytes = Cache.getMemoryUsed() - EmptyBytes;
    ASSERT_GT(LargeBytes, 8 * Large.size() * sizeof(float));

    for (uint64_t H = 9; H <= 16; ++H) {
        ASSERT_TRUE(Cache.store(H, (uint16_t)Small.size(), Small.data(),
                                0.75f, 0.125f));
    }
    EXPECT_LT(Cache.getMemoryUsed() - EmptyBytes, LargeBytes / 4);
    EvalCache::EvalInfo EI;
    for (uint64_t H = 9; H <= 16; ++H) {
        ASSERT_TRUE(Cache.load(H, &EI));
        ASSERT_EQ(EI.Policy, Small);
    }
}

TEST(EvalCache, DynamicStorageRespectsMemoryBudget) {
    EvalCache Cache(1);
    const std::vector<float> Policy(4096, 0.5f);
    EvalCache::EvalInfo EI;
    for (uint64_t I = 1; I <= 2000; ++I) {
        const uint64_t Hash = I * 0x9E3779B97F4A7C15ULL;
        ASSERT_TRUE(Cache.store(Hash, (uint16_t)Policy.size(), Policy.data(),
                                0.75f, 0.125f));
        ASSERT_LE(Cache.getMemoryUsed(), Cache.getMemoryLimit());
        ASSERT_TRUE(Cache.load(Hash, &EI));
        ASSERT_EQ(EI.Policy, Policy);
    }
}

TEST(EvalCache, EntryLargerThanMemoryBudgetIsRejected) {
    EvalCache Cache(0);
    const std::vector<float> Policy(UINT16_MAX, 0.5f);
    const std::size_t Before = Cache.getMemoryUsed();
    ASSERT_FALSE(Cache.store(1, UINT16_MAX, Policy.data(), 0.75f, 0.125f));
    EXPECT_EQ(Cache.getMemoryUsed(), Before);
}

TEST(EvalCache, PoliciesBeyondCompactOffsetsRemainLoadable) {
    EvalCache Cache(4);
    std::vector<float> Policy(UINT16_MAX);
    for (uint64_t H = 1; H <= 8; ++H) {
        std::fill(Policy.begin(), Policy.end(), (float)H);
        ASSERT_TRUE(Cache.store(H, UINT16_MAX, Policy.data(), 0.75f, 0.125f));
    }
    EvalCache::EvalInfo EI;
    for (uint64_t H = 1; H <= 8; ++H) {
        ASSERT_TRUE(Cache.load(H, &EI));
        ASSERT_EQ(EI.Policy.size(), (std::size_t)UINT16_MAX);
        for (float Value : EI.Policy) {
            ASSERT_EQ(Value, (float)H);
        }
    }
}

TEST(EvalCache, ConcurrentReclamationKeepsEntriesIntact) {
    EvalCache Cache(1);
    constexpr int NumThreads = 8;
    std::barrier Start(NumThreads);
    std::atomic<uint64_t> NumHits{0};
    std::vector<std::thread> Threads;
    for (int T = 0; T < NumThreads; ++T) {
        Threads.emplace_back([&Cache, &Start, &NumHits, T]() {
            std::mt19937_64 Rng((uint64_t)T + 42);
            EvalCache::EvalInfo EI;
            Start.arrive_and_wait();
            for (int I = 0; I < 20000; ++I) {
                // Spread keys over the entire table. Policies exceed the
                // memory budget together, forcing cross-bucket reclamation
                // while other threads read and resize their own buckets.
                const uint64_t H = (1 + Rng() % 1024) * 0x9E3779B97F4A7C15ULL;
                if (T < NumThreads / 2) {
                    fillEntry(H, &EI);
                    Cache.store(H, EI.NumMoves, EI.Policy.data(), EI.WinRate,
                                EI.DrawRate);
                }
                if (Cache.load(H, &EI)) {
                    expectEntryMatches(H, EI);
                    NumHits.fetch_add(1, std::memory_order_relaxed);
                }
                ASSERT_LE(Cache.getMemoryUsed(), Cache.getMemoryLimit());
            }
        });
    }
    for (auto& Thread : Threads) {
        Thread.join();
    }
    ASSERT_GT(NumHits.load(), 0ULL);
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
                Cache.store(H, EI.NumMoves, EI.Policy.data(), EI.WinRate, EI.DrawRate);
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
