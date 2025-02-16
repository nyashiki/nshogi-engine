//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../math/fixedpoint.h"
#include <gtest/gtest.h>
#include <random>

TEST(Math, FixedPointHandmade1) {
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    EXPECT_NEAR(Temp.toDouble(), 0.0, 1e-12);
}

TEST(Math, FixedPointHandmade2) {
    nshogi::engine::math::FixedPoint64 Temp(1.0);

    EXPECT_NEAR(Temp.toDouble(), 1.0, 1e-12);
}

TEST(Math, FixedPointFloatHandmade2) {
    nshogi::engine::math::FixedPoint64 Temp(0.5f);

    EXPECT_NEAR(Temp.toDouble(), 0.5, 1e-12);
}

TEST(Math, FixedPointArange) {
    for (double D = 0; D < 1.0; D += 1e-6) {
        nshogi::engine::math::FixedPoint64 Temp(D);

        EXPECT_NEAR(Temp.toDouble(), D, 1e-10);
    }
}

TEST(Math, FixedPointAddHandmade1) {
    nshogi::engine::math::FixedPoint64 Temp(0.0);
    Temp.add(0.0);

    EXPECT_NEAR(Temp.toDouble(), 0.0, 1e-12);
}

TEST(Math, FixedPointAddHandmade2) {
    nshogi::engine::math::FixedPoint64 Temp(0.5);
    Temp.add(0.5);

    EXPECT_NEAR(Temp.toDouble(), 1.0, 1e-12);
}

TEST(Math, FixedPointAddHandmade3) {
    nshogi::engine::math::FixedPoint64 Temp(0.1234);
    Temp.add(0.4321);

    EXPECT_NEAR(Temp.toDouble(), 0.5555, 1e-10);
}

TEST(Math, FixedPointAddOne100) {
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    for (int I = 0; I < 100; ++I) {
        Temp.addOne();
    }

    EXPECT_NEAR(Temp.toDouble(), 100.0, 1e-10);
}

TEST(Math, FixedPointAddRandom) {
    std::mt19937_64 Mt(20230829);
    std::uniform_real_distribution<double> Distribution(0.0, 1.0);

    const int N = 10000000;

    double Sum = 0;
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    for (int I = 0; I < N; ++I) {
        double R = Distribution(Mt);

        Sum += R;
        Temp.add(R);
    }

    EXPECT_NEAR(Sum, Temp.toDouble(), 1e-4);
}

TEST(Math, FixedPointAddRandomTiny) {
    std::mt19937_64 Mt(20230829);
    std::uniform_real_distribution<double> Distribution(0.0, 0.01);

    const int N = 10000000;

    double Sum = 0;
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    for (int I = 0; I < N; ++I) {
        double R = Distribution(Mt);

        Sum += R;
        Temp.add(R);
    }

    EXPECT_NEAR(Sum, Temp.toDouble(), 1e-4);
}

TEST(Math, FixedPointMulHandmade1) {
    nshogi::engine::math::FixedPoint64 Temp1(0.0);
    nshogi::engine::math::FixedPoint64 Temp2(0.0);

    auto Temp = Temp1 * Temp2;

    EXPECT_NEAR(Temp.toDouble(), 0.0, 1e-12);
}

TEST(Math, FixedPointMulHandmade2) {
    nshogi::engine::math::FixedPoint64 Temp1(0.5);
    nshogi::engine::math::FixedPoint64 Temp2(0.5);

    auto Temp = Temp1 * Temp2;

    EXPECT_NEAR(Temp.toDouble(), 0.25, 1e-12);
}

TEST(Math, FixedPointMulHandmade3) {
    nshogi::engine::math::FixedPoint64 Temp1(0.1234);
    nshogi::engine::math::FixedPoint64 Temp2(0.9876);

    auto Temp = Temp1 * Temp2;

    EXPECT_NEAR(Temp.toDouble(), 0.12186984, 1e-5);
}
