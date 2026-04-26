//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../cuda/extractbit.h"
#include "../globalconfig.h"
#include "cuda_runtime.h"
#include <cstdio>
#include <nshogi/core/movegenerator.h>
#include <nshogi/core/positionbuilder.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/core/types.h>
#include <nshogi/ml/featurestack.h>
#include <random>
#include <vector>

namespace {

template <bool ChannelsFirst>
void testExtractBit(const nshogi::core::State& State) {
    using namespace nshogi::ml;
    using FeatureType = nshogi::engine::global_config::FeatureType;

    nshogi::core::StateConfig Config;
    Config.MaxPly = 1024;
    Config.Rule = nshogi::core::ER_Declare27;

    FeatureType FS(State, Config);

    uint64_t* DeviceIn;
    float* DeviceOut;

    cudaMalloc(&DeviceIn, FeatureType::size() * 2 * sizeof(uint64_t));
    cudaMalloc(&DeviceOut, FeatureType::size() * 81 * sizeof(float));

    cudaMemcpy(DeviceIn, (const void*)FS.data(),
               FeatureType::size() * 2 * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    nshogi::engine::cuda::extractBits<ChannelsFirst>(DeviceOut, DeviceIn, 1,
                                                     FeatureType::size());

    std::vector<float> Output(FeatureType::size() * 81);
    cudaMemcpy(Output.data(), DeviceOut,
               FeatureType::size() * 81 * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(DeviceOut);
    cudaFree(DeviceIn);

    std::vector<float> Expected =
        FS.extract<nshogi::core::IterateOrder::ESWN, ChannelsFirst>();

    for (std::size_t I = 0; I < FeatureType::size() * 81; ++I) {
        EXPECT_NEAR(Expected[I], Output[I], 1e-4);
    }
}

} // namespace

TEST(ExtractBit, RandomStates) {
    const int N = 1;
    const int PlyMax = 1024;

    std::mt19937_64 Mt(20240203);

    for (int Count = 0; Count < N; ++Count) {
        auto State = nshogi::core::StateBuilder::getInitialState();
        for (int Ply = 0; Ply < PlyMax; ++Ply) {
            const auto Moves =
                nshogi::core::MoveGenerator::generateLegalMoves(State);

            if (Moves.size() == 0) {
                break;
            }

            const auto RandomMove = Moves[Mt() % Moves.size()];
            State.doMove(RandomMove);

            testExtractBit<true>(State);
            testExtractBit<false>(State);
        }
    }
}
