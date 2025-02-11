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

#include "protocol/usi.h"

#include <nshogi/core/initializer.h>

int main() {
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
