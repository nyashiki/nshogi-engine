//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>
#include <nshogi/core/initializer.h>

int main(int argc, char *argv[]) {
    nshogi::core::initializer::initializeAll();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
