//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../allocator/fixed_allocator.h"
#include "../allocator/segregated_free_list.h"

#include <random>

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

TEST(SegregatedAllocator, RandomWrite) {
    std::mt19937_64 mt(20231219);

    constexpr std::size_t Size = 100000;
    constexpr std::size_t N = 100000000;

    std::vector<int> Vec(Size, 0);
    std::vector<int*> Vec2(Size, nullptr);

    nshogi::engine::allocator::SegregatedFreeListAllocator Allocator;
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
}

TEST(SegregatedAllocator, DifferentSizes) {
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

    nshogi::engine::allocator::SegregatedFreeListAllocator Allocator;
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
}
