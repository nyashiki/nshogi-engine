//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../allocator/fixed_allocator.h"
#include "../allocator/slab.h"

#include <atomic>
#include <cstring>
#include <random>
#include <thread>

TEST(FixedAllocator, RandomWrite) {
    std::mt19937_64 mt(20231219);

    constexpr std::size_t Size = 100000;
    constexpr std::size_t N = 100000000;

    std::vector<int> Vec(Size, 0);
    std::vector<int*> Vec2(Size, nullptr);

    nshogi::engine::allocator::FixedAllocator<sizeof(int)> Allocator;
    Allocator.resize(1ULL * 1024 * 1024 * 1024);

    for (std::size_t I = 0; I < Size; ++I) {
        Vec2[I] = reinterpret_cast<int*>(Allocator.malloc(0));
        *Vec2[I] = 0;
    }

    for (std::size_t I = 0; I < N; ++I) {
        const std::size_t R1 = mt() % Size;
        const int R2 = (int)mt();

        Vec[R1] = R2;
        *Vec2[R1] = R2;

        ASSERT_EQ(Vec[R1], *Vec2[R1]);
    }

    for (std::size_t I = 0; I < Size; ++I) {
        ASSERT_EQ(Vec[I], *Vec2[I]);
    }

    for (std::size_t I = 0; I < Size; ++I) {
        ASSERT_EQ(Vec[I], *Vec2[I]);
        Allocator.free(Vec2[I]);
    }
}

TEST(SlabAllocator, RandomWrite) {
    std::mt19937_64 mt(20231219);

    constexpr std::size_t Size = 100000;
    constexpr std::size_t N = 100000000;

    std::vector<int> Vec(Size, 0);
    std::vector<int*> Vec2(Size, nullptr);

    nshogi::engine::allocator::SlabAllocator Allocator;
    Allocator.resize(1ULL * 1024 * 1024 * 1024);

    for (std::size_t I = 0; I < Size; ++I) {
        Vec2[I] = reinterpret_cast<int*>(Allocator.malloc(sizeof(int)));
        *Vec2[I] = 0;
    }

    for (std::size_t I = 0; I < N; ++I) {
        const std::size_t R1 = mt() % Size;
        const int R2 = (int)mt();

        Vec[R1] = R2;
        *Vec2[R1] = R2;

        ASSERT_EQ(Vec[R1], *Vec2[R1]);
    }

    for (std::size_t I = 0; I < Size; ++I) {
        ASSERT_EQ(Vec[I], *Vec2[I]);
    }

    for (std::size_t I = 0; I < Size; ++I) {
        ASSERT_EQ(Vec[I], *Vec2[I]);
        Allocator.free(Vec2[I]);
    }

    ASSERT_TRUE(Allocator.isAllBlockFree());
}

TEST(SlabAllocator, DifferentSizes) {
    std::mt19937_64 mt(20231219);

    constexpr std::size_t Size = 10000;
    constexpr std::size_t N = 100000000;

    std::vector<int8_t> VecI8(Size, 0);
    std::vector<int16_t> VecI16(Size, 0);
    std::vector<int32_t> VecI32(Size, 0);
    std::vector<int64_t> VecI64(Size, 0);

    std::vector<int8_t*> VecI8_(Size, nullptr);
    std::vector<int16_t*> VecI16_(Size, nullptr);
    std::vector<int32_t*> VecI32_(Size, nullptr);
    std::vector<int64_t*> VecI64_(Size, nullptr);

    nshogi::engine::allocator::SlabAllocator Allocator;
    Allocator.resize(1ULL * 1024 * 1024 * 1024);

    for (std::size_t I = 0; I < Size; ++I) {
        VecI8_[I] = reinterpret_cast<int8_t*>(Allocator.malloc(sizeof(int8_t)));
        VecI16_[I] =
            reinterpret_cast<int16_t*>(Allocator.malloc(sizeof(int16_t)));
        VecI32_[I] =
            reinterpret_cast<int32_t*>(Allocator.malloc(sizeof(int32_t)));
        VecI64_[I] =
            reinterpret_cast<int64_t*>(Allocator.malloc(sizeof(int64_t)));

        *VecI8_[I] = 0;
        *VecI16_[I] = 0;
        *VecI32_[I] = 0;
        *VecI64_[I] = 0;
    }

    for (std::size_t I = 0; I < N; ++I) {
        const std::size_t Index1 = mt() % Size;
        const std::size_t Index2 = mt() % Size;
        const std::size_t Index3 = mt() % Size;
        const std::size_t Index4 = mt() % Size;
        const int8_t R1 = (int8_t)mt();
        const int16_t R2 = (int16_t)mt();
        const int32_t R3 = (int32_t)mt();
        const int64_t R4 = (int64_t)mt();

        VecI8[Index1] = R1;
        VecI16[Index2] = R2;
        VecI32[Index3] = R3;
        VecI64[Index4] = R4;

        *VecI8_[Index1] = R1;
        *VecI16_[Index2] = R2;
        *VecI32_[Index3] = R3;
        *VecI64_[Index4] = R4;

        ASSERT_EQ(VecI8[Index1], *VecI8_[Index1]);
        ASSERT_EQ(VecI16[Index2], *VecI16_[Index2]);
        ASSERT_EQ(VecI32[Index3], *VecI32_[Index3]);
        ASSERT_EQ(VecI64[Index4], *VecI64_[Index4]);
    }

    for (std::size_t I = 0; I < Size; ++I) {
        ASSERT_EQ(VecI8[I], *VecI8_[I]);
        ASSERT_EQ(VecI16[I], *VecI16_[I]);
        ASSERT_EQ(VecI32[I], *VecI32_[I]);
        ASSERT_EQ(VecI64[I], *VecI64_[I]);

        Allocator.free(VecI8_[I]);
        Allocator.free(VecI16_[I]);
        Allocator.free(VecI32_[I]);
        Allocator.free(VecI64_[I]);
    }

    ASSERT_TRUE(Allocator.isAllBlockFree());
}

TEST(SlabAllocator, EmptySlabRecyclingAcrossClasses) {
    std::mt19937_64 mt(20260716);

    nshogi::engine::allocator::SlabAllocator Allocator;
    Allocator.resize(64ULL << 20);

    // Fill with one class, free everything, then the whole arena must
    // be servable by a different class.
    for (int Phase = 0; Phase < 4; ++Phase) {
        const std::size_t Bytes = 16 * (20 + 60 * (std::size_t)Phase);

        std::vector<void*> Live;
        while (true) {
            void* P = Allocator.malloc(Bytes);
            if (P == nullptr) {
                break;
            }
            Live.push_back(P);
        }

        // At least ~95% of the arena must have been usable in every
        // phase; slabs freed by earlier phases must not be stranded.
        ASSERT_GT((double)(Live.size() * Bytes),
                  0.95 * (double)Allocator.getTotal());

        for (void* P : Live) {
            Allocator.free(P);
        }

        ASSERT_TRUE(Allocator.isAllBlockFree());
        ASSERT_EQ(Allocator.getUsed(), 0ULL);
    }
}

TEST(SlabAllocator, MultiThreadedChurn) {
    nshogi::engine::allocator::SlabAllocator Allocator;
    Allocator.resize(1ULL * 1024 * 1024 * 1024);

    constexpr std::size_t NumThreads = 8;
    constexpr std::size_t Slots = 5000;
    constexpr std::size_t N = 200000;

    std::atomic<bool> Failed(false);
    std::vector<std::thread> Threads;

    for (std::size_t T = 0; T < NumThreads; ++T) {
        Threads.emplace_back([&Allocator, &Failed, T]() {
            std::mt19937_64 mt(20260719 + T);

            struct Rec {
                uint8_t* P = nullptr;
                std::size_t Bytes = 0;
                uint8_t Tag = 0;
            };
            std::vector<Rec> Live(Slots);

            const auto verify = [](const Rec& R) {
                for (std::size_t I = 0; I < R.Bytes; ++I) {
                    if (R.P[I] != R.Tag) {
                        return false;
                    }
                }
                return true;
            };

            for (std::size_t Op = 0; Op < N; ++Op) {
                Rec& R = Live[mt() % Slots];
                if (R.P != nullptr) {
                    if (!verify(R)) {
                        Failed.store(true);
                        return;
                    }
                    Allocator.free(R.P);
                }
                R.Bytes = 16 * (1 + mt() % 593);
                R.P = reinterpret_cast<uint8_t*>(Allocator.malloc(R.Bytes));
                if (R.P == nullptr) {
                    Failed.store(true);
                    return;
                }
                R.Tag = (uint8_t)(mt());
                std::memset(R.P, R.Tag, R.Bytes);
            }

            for (Rec& R : Live) {
                if (R.P != nullptr) {
                    if (!verify(R)) {
                        Failed.store(true);
                        return;
                    }
                    Allocator.free(R.P);
                }
            }
        });
    }

    for (auto& T : Threads) {
        T.join();
    }

    ASSERT_FALSE(Failed.load());
    ASSERT_TRUE(Allocator.isAllBlockFree());
    ASSERT_EQ(Allocator.getUsed(), 0ULL);
}

TEST(SlabAllocator, RuntimeShardCount) {
    for (const std::size_t NumShards : {1UL, 3UL, 16UL}) {
        nshogi::engine::allocator::SlabAllocator Allocator;
        Allocator.setNumShards(NumShards);
        Allocator.resize(256ULL * 1024 * 1024);
        ASSERT_EQ(Allocator.getNumShards(), NumShards);

        constexpr std::size_t NumThreads = 8;
        constexpr std::size_t Slots = 1000;
        constexpr std::size_t N = 30000;

        std::atomic<bool> Failed(false);
        std::vector<std::thread> Threads;
        for (std::size_t T = 0; T < NumThreads; ++T) {
            Threads.emplace_back([&Allocator, &Failed, T]() {
                std::mt19937_64 mt(T);
                std::vector<std::pair<uint8_t*, std::size_t>> Live(Slots);
                for (std::size_t Op = 0; Op < N; ++Op) {
                    auto& R = Live[mt() % Slots];
                    if (R.first != nullptr) {
                        Allocator.free(R.first);
                    }
                    R.second = 16 * (1 + mt() % 593);
                    R.first =
                        reinterpret_cast<uint8_t*>(Allocator.malloc(R.second));
                    if (R.first == nullptr) {
                        Failed.store(true);
                        return;
                    }
                    std::memset(R.first, (int)T, R.second);
                }
                for (auto& R : Live) {
                    if (R.first != nullptr) {
                        Allocator.free(R.first);
                    }
                }
            });
        }
        for (auto& T : Threads) {
            T.join();
        }

        ASSERT_FALSE(Failed.load());
        ASSERT_TRUE(Allocator.isAllBlockFree());
        ASSERT_EQ(Allocator.getUsed(), 0ULL);
    }
}
