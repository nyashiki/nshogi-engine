//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

int setupExtractBit();
int setupMath();
int setupCudaMath();
int setupAllocator();

int main() {
    CU_initialize_registry();

    setupExtractBit();
    setupMath();
    setupCudaMath();
    setupAllocator();

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();

    CU_cleanup_registry();

    return CU_get_error();
}
