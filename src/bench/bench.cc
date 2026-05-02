//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "batchsize.h"
#include "mcts.h"

#include <nshogi/core/initializer.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& S) {
    std::vector<std::string> Result;
    std::stringstream SS(S);
    std::string Part;

    while (std::getline(SS, Part, ' ')) {
        Result.push_back(Part);
    }

    return Result;
}

} // namespace

int main() {
    using namespace nshogi::engine;
    using namespace nshogi::engine::bench;

    nshogi::core::initializer::initializeAll();

    std::cout << "Entering bench mode." << std::endl;

    std::string Line;
    while (std::getline(std::cin, Line)) {
        // Split the input line into command and arguments.
        const auto Split = split(Line);

        if (Split[0] == "BatchSize") {
            std::string WeightPath = "./res/model.onnx";
            std::size_t Repeat = 3000;

            if (Split.size() >= 2) {
                WeightPath = Split[1];
            }

            if (Split.size() >= 3) {
                Repeat = std::stoul(Split[2]);
            }

            benchBatchSize(WeightPath.c_str(), Repeat);
        } else if (Split[0] == "MCTS") {
            const uint64_t DurationSeconds =
                (Split.size() >= 2) ? std::stoull(Split[1]) : 10;
            const std::size_t BatchSize =
                (Split.size() >= 3) ? std::stoul(Split[2]) : 128;
            const std::size_t NumGPUs =
                (Split.size() >= 4) ? std::stoul(Split[3]) : 1;
            const std::size_t NumThreadsPerGPU =
                (Split.size() >= 5) ? std::stoul(Split[4]) : 1;
            const std::size_t NumCheckmateSearchers =
                (Split.size() >= 6) ? std::stoul(Split[5]) : 0;
            const std::size_t EvalCacheMB =
                (Split.size() >= 7) ? std::stoul(Split[6]) : 0;

            benchMCTS(DurationSeconds, BatchSize, NumGPUs, NumThreadsPerGPU,
                      NumCheckmateSearchers, EvalCacheMB);
        } else if (Split[0] == "quit" || Split[0] == "exit") {
            break;
        } else {
            std::cout << "Unknown command `" << Split[0] << "`."
                      << std::endl;
        }
    }

    return 0;
}
