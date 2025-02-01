//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <CUnit/CUnit.h>

#include "../math/fixedpoint.h"
#include <vector>
#include <iostream>
#include <random>

namespace {

void testFixedPointHandmade1() {
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.0, 1e-12);
}

void testFixedPointHandmade2() {
    nshogi::engine::math::FixedPoint64 Temp(1.0);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 1.0, 1e-12);
}

void testFixedPointFloatHandmade1() {
    nshogi::engine::math::FixedPoint64 Temp(0.5f);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.5, 1e-12);
}

void testFixedPointArange() {
    for (double D = 0; D < 1.0; D += 1e-6) {
        nshogi::engine::math::FixedPoint64 Temp(D);

        CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), D, 1e-10);
    }
}

void testFixedPointAddHandmade1() {
    nshogi::engine::math::FixedPoint64 Temp(0.0);
    Temp.add(0.0);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.0, 1e-12);
}

void testFixedPointAddHandmade2() {
    nshogi::engine::math::FixedPoint64 Temp(0.5);
    Temp.add(0.5);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 1.0, 1e-12);
}

void testFixedPointAddHandmade3() {
    nshogi::engine::math::FixedPoint64 Temp(0.1234);
    Temp.add(0.4321);

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.5555, 1e-10);
}

void testFixedPointAddOne100() {
    nshogi::engine::math::FixedPoint64 Temp(0.0);

    for (int I = 0; I < 100; ++I) {
        Temp.addOne();
    }

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 100.0, 1e-10);
}

void testFixedPointAddRandom() {
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

    CU_ASSERT_DOUBLE_EQUAL(Sum, Temp.toDouble(), 1e-4);
}

void testFixedPointAddRandomTiny() {
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

    CU_ASSERT_DOUBLE_EQUAL(Sum, Temp.toDouble(), 1e-4);
}

void testFixedPointMulHandmade1() {
    nshogi::engine::math::FixedPoint64 Temp1(0.0);
    nshogi::engine::math::FixedPoint64 Temp2(0.0);

    auto Temp = Temp1 * Temp2;

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.0, 1e-12);
}

void testFixedPointMulHandmade2() {
    nshogi::engine::math::FixedPoint64 Temp1(0.5);
    nshogi::engine::math::FixedPoint64 Temp2(0.5);

    auto Temp = Temp1 * Temp2;

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.25, 1e-12);
}

void testFixedPointMulHandmade3() {
    nshogi::engine::math::FixedPoint64 Temp1(0.1234);
    nshogi::engine::math::FixedPoint64 Temp2(0.9876);

    auto Temp = Temp1 * Temp2;

    CU_ASSERT_DOUBLE_EQUAL(Temp.toDouble(), 0.12186984, 1e-5);
}


}


int setupMath() {
    CU_pSuite suite = CU_add_suite("math test", 0, 0);

    CU_add_test(suite, "testFixedPoint Handmade1", testFixedPointHandmade1);
    CU_add_test(suite, "testFixedPoint Handmade2", testFixedPointHandmade2);
    CU_add_test(suite, "testFixedPoint Float Handmade1", testFixedPointFloatHandmade1);
    CU_add_test(suite, "testFixedPoint Arange", testFixedPointArange);
    CU_add_test(suite, "testFixedPoint Add Handmade 1", testFixedPointAddHandmade1);
    CU_add_test(suite, "testFixedPoint Add Handmade 2", testFixedPointAddHandmade2);
    CU_add_test(suite, "testFixedPoint Add Handmade 3", testFixedPointAddHandmade3);
    CU_add_test(suite, "testFixedPoint Add One 100", testFixedPointAddOne100);
    CU_add_test(suite, "testFixedPoint Add Random", testFixedPointAddRandom);
    CU_add_test(suite, "testFixedPoint Add Random Tiny", testFixedPointAddRandomTiny);
    CU_add_test(suite, "testFixedPoint Mul Handmade 1", testFixedPointMulHandmade1);
    CU_add_test(suite, "testFixedPoint Mul Handmade 2", testFixedPointMulHandmade2);
    CU_add_test(suite, "testFixedPoint Mul Handmade 3", testFixedPointMulHandmade3);

    return CUE_SUCCESS;
}
