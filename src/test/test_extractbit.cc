//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../cuda/extractbit.h"

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

void testExtractBit(const nshogi::core::State& State) {
    using namespace nshogi::ml;
    using FeatureTypes = nshogi::ml::FeatureStackComptime<
        FeatureType::FT_Black, FeatureType::FT_White, FeatureType::FT_MyPawn,
        FeatureType::FT_MyLance, FeatureType::FT_MyKnight,
        FeatureType::FT_MySilver, FeatureType::FT_MyGold,
        FeatureType::FT_MyKing, FeatureType::FT_MyBishop,
        FeatureType::FT_MyRook, FeatureType::FT_MyProPawn,
        FeatureType::FT_MyProLance, FeatureType::FT_MyProKnight,
        FeatureType::FT_MyProSilver, FeatureType::FT_MyProBishop,
        FeatureType::FT_MyProRook, FeatureType::FT_OpPawn,
        FeatureType::FT_OpLance, FeatureType::FT_OpKnight,
        FeatureType::FT_OpSilver, FeatureType::FT_OpGold,
        FeatureType::FT_OpKing, FeatureType::FT_OpBishop,
        FeatureType::FT_OpRook, FeatureType::FT_OpProPawn,
        FeatureType::FT_OpProLance, FeatureType::FT_OpProKnight,
        FeatureType::FT_OpProSilver, FeatureType::FT_OpProBishop,
        FeatureType::FT_OpProRook, FeatureType::FT_MyStandPawn1,
        FeatureType::FT_MyStandPawn2, FeatureType::FT_MyStandPawn3,
        FeatureType::FT_MyStandPawn4, FeatureType::FT_MyStandPawn5,
        FeatureType::FT_MyStandPawn6, FeatureType::FT_MyStandPawn7,
        FeatureType::FT_MyStandPawn8, FeatureType::FT_MyStandPawn9,
        FeatureType::FT_MyStandLance1, FeatureType::FT_MyStandLance2,
        FeatureType::FT_MyStandLance3, FeatureType::FT_MyStandLance4,
        FeatureType::FT_MyStandKnight1, FeatureType::FT_MyStandKnight2,
        FeatureType::FT_MyStandKnight3, FeatureType::FT_MyStandKnight4,
        FeatureType::FT_MyStandSilver1, FeatureType::FT_MyStandSilver2,
        FeatureType::FT_MyStandSilver3, FeatureType::FT_MyStandSilver4,
        FeatureType::FT_MyStandGold1, FeatureType::FT_MyStandGold2,
        FeatureType::FT_MyStandGold3, FeatureType::FT_MyStandGold4,
        FeatureType::FT_MyStandBishop1, FeatureType::FT_MyStandBishop2,
        FeatureType::FT_MyStandRook1, FeatureType::FT_MyStandRook2,
        FeatureType::FT_OpStandPawn1, FeatureType::FT_OpStandPawn2,
        FeatureType::FT_OpStandPawn3, FeatureType::FT_OpStandPawn4,
        FeatureType::FT_OpStandPawn5, FeatureType::FT_OpStandPawn6,
        FeatureType::FT_OpStandPawn7, FeatureType::FT_OpStandPawn8,
        FeatureType::FT_OpStandPawn9, FeatureType::FT_OpStandLance1,
        FeatureType::FT_OpStandLance2, FeatureType::FT_OpStandLance3,
        FeatureType::FT_OpStandLance4, FeatureType::FT_OpStandKnight1,
        FeatureType::FT_OpStandKnight2, FeatureType::FT_OpStandKnight3,
        FeatureType::FT_OpStandKnight4, FeatureType::FT_OpStandSilver1,
        FeatureType::FT_OpStandSilver2, FeatureType::FT_OpStandSilver3,
        FeatureType::FT_OpStandSilver4, FeatureType::FT_OpStandGold1,
        FeatureType::FT_OpStandGold2, FeatureType::FT_OpStandGold3,
        FeatureType::FT_OpStandGold4, FeatureType::FT_OpStandBishop1,
        FeatureType::FT_OpStandBishop2, FeatureType::FT_OpStandRook1,
        FeatureType::FT_OpStandRook2, FeatureType::FT_Check,
        FeatureType::FT_NoMyPawnFile, FeatureType::FT_NoOpPawnFile,
        FeatureType::FT_Progress, FeatureType::FT_ProgressUnit,
        FeatureType::FT_RuleDeclare27, FeatureType::FT_RuleDraw24,
        FeatureType::FT_RuleTrying, FeatureType::FT_MyDrawValue,
        FeatureType::FT_OpDrawValue, FeatureType::FT_MyDeclarationScore,
        FeatureType::FT_OpDeclarationScore, FeatureType::FT_MyPieceScore,
        FeatureType::FT_OpPieceScore>;

    nshogi::core::StateConfig Config;
    Config.MaxPly = 1024;
    Config.Rule = nshogi::core::ER_Declare27;

    FeatureTypes FS(State, Config);

    uint64_t* DeviceIn;
    float* DeviceOut;

    cudaMalloc(&DeviceIn, FeatureTypes::size() * 2 * sizeof(uint64_t));
    cudaMalloc(&DeviceOut, FeatureTypes::size() * 81 * sizeof(float));

    cudaMemcpy(DeviceIn, (const void*)FS.data(),
               FeatureTypes::size() * 2 * sizeof(uint64_t),
               cudaMemcpyHostToDevice);

    nshogi::engine::cuda::extractBits(DeviceOut, DeviceIn,
                                      FeatureTypes::size());

    std::vector<float> Output(FeatureTypes::size() * 81);
    cudaMemcpy(Output.data(), DeviceOut,
               FeatureTypes::size() * 81 * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(DeviceOut);
    cudaFree(DeviceIn);

    std::vector<float> Expected =
        FS.extract<nshogi::core::IterateOrder::ESWN>();

    for (std::size_t I = 0; I < FeatureTypes::size() * 81; ++I) {
        EXPECT_NEAR(Expected[I], Output[I], 1e-4);
    }
}

} // namespace

TEST(ExtractBit, RandomStates) {
    const int N = 100;
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

            testExtractBit(State);
        }
    }
}
