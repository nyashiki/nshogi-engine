//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "allocator/allocator.h"
#include "allocator/fixed_allocator.h"
#include "allocator/segregated_free_list.h"
#include "mcts/node.h"
#include "mcts/edge.h"

#include <cstdio>
#include <cinttypes>
#include "protocol/usi.h"

#include <nshogi/core/initializer.h>

int main() {
    // nshogi::engine::allocator::SegregatedFreeListAllocator<128> Alloc;
    // // nshogi::engine::allocator::FixedAllocator<8> Alloc;
    // Alloc.resize(1024);
    // void* Ptr1 = Alloc.malloc(10);
    // void* Ptr2 = Alloc.malloc(10);
    // Alloc.free(Ptr1);
    // Alloc.free(Ptr2);
    // return 0;

    nshogi::core::initializer::initializeAll();

    std::string Command;

    while (std::cin >> Command) {
        if (Command == "usi") {
            nshogi::engine::protocol::usi::mainLoop();
            break;
        } else if (Command == "quit" || Command == "exit") {
            break;
        } else {
            std::cout << "Unknown command `" << Command << "`." << std::endl;
        }
    }

    return 0;
}
