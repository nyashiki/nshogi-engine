#include <CUnit/CUnit.h>

#include "../cuda/extractbit.h"
#include <vector>
#include <iostream>
#include "cuda_runtime.h"
#include <cstdio>
#include <nshogi/core/bitboard.h>
#include <nshogi/core/types.h>
#include <nshogi/ml/featurebitboard.h>

namespace {

void testOneSquare(nshogi::core::Square Sq, float Value) {
    nshogi::ml::FeatureBitboard FB(nshogi::core::bitboard::SquareBB[Sq], Value, false);

    uint64_t* DeviceIn;
    float* DeviceOut;

    cudaMalloc(&DeviceIn, 2 * sizeof(uint64_t));
    cudaMalloc(&DeviceOut, 81 * sizeof(float));

    cudaMemcpy(DeviceIn, (const void*)FB.data(), 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    nshogi::engine::cuda::extractBits(DeviceOut, DeviceIn, 1);

    std::vector<float> Output(81);
    cudaMemcpy(Output.data(), DeviceOut, 81 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(DeviceOut);
    cudaFree(DeviceIn);

    for (auto S : nshogi::core::Squares) {
        if (S == Sq) {
            CU_ASSERT_DOUBLE_EQUAL(Output[std::size_t(S)], Value, 1e-6);
        } else {
            CU_ASSERT_DOUBLE_EQUAL(Output[std::size_t(S)], 0, 1e-6);
        }
    }
}

void testOneSquares() {
    for (float Value = -10.0f; Value < 10.0f; Value += 0.1f) {
        for (auto Sq : nshogi::core::Squares) {
            testOneSquare(Sq, Value);
        }
    }
}


} // namespace


int setupExtractBit() {
    CU_pSuite suite = CU_add_suite("extractbit test", 0, 0);

    CU_add_test(suite, "testOneSquares", testOneSquares);

    return CUE_SUCCESS;
}
