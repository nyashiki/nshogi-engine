//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "batchsize.h"
#include "mcts.h"

#include <nshogi/core/utils.h>
#include <nshogi/core/initializer.h>

#include <iostream>
#include <string>
#include <vector>

int main(int Argc, char* Argv[]) {
    using namespace nshogi::engine;
    using namespace nshogi::engine::bench;

    nshogi::core::initializer::initializeAll();

    std::cout << "Entering bench mode." << std::endl;

    std::string Line;
    while (Argc >= 2 || std::getline(std::cin, Line)) {
        if (Argc == 1 && Line.size() == 0) {
            continue;
        }

        const auto Splitted = (Argc >= 2)
            ? std::vector<std::string>(Argv + 1, Argv + Argc)
            : nshogi::core::utils::split(Line, ' ');

        if (Splitted[0] == "BatchSize") {
            std::string WeightPath = "./res/model.onnx";
            std::size_t Repeat = 1000;

            if (Splitted.size() >= 2) {
                WeightPath = Splitted[1];
            }

            if (Splitted.size() >= 3) {
                Repeat = std::stoul(Splitted[2]);
            }

            benchBatchSize(WeightPath.c_str(), Repeat);
        } else if (Splitted[0] == "MCTS") {
            const uint64_t    DurationSeconds       = (Splitted.size() >= 2) ? std::stoull(Splitted[1]) :  10;
            const std::size_t BatchSize             = (Splitted.size() >= 3) ? std::stoul(Splitted[2])  : 128;
            const std::size_t NumGPUs               = (Splitted.size() >= 4) ? std::stoul(Splitted[3])  :   1;
            const std::size_t NumThreadsPerGPU      = (Splitted.size() >= 5) ? std::stoul(Splitted[4])  :   1;
            const std::size_t NumCheckmateSearchers = (Splitted.size() >= 6) ? std::stoul(Splitted[5])  :   0;
            const std::size_t EvalCacheMB           = (Splitted.size() >= 7) ? std::stoul(Splitted[6])  :   0;

            benchMCTS(DurationSeconds, BatchSize, NumGPUs, NumThreadsPerGPU, NumCheckmateSearchers, EvalCacheMB);
        } else if (Splitted[0] == "quit" || Splitted[0] == "exit") {
            break;
        } else {
            std::cout << "Unknown command `" << Splitted[0] << "`." << std::endl;
        }

        if (Argc >= 2) {
            break;
        }
    }

    return 0;
}
