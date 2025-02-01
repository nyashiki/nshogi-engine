//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <CUnit/CUnit.h>

#include "../cuda/math.h"

#include <functional>
#include <random>
#include <vector>

namespace {

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void testSigmoid() {
    std::size_t N = 10000;
    std::size_t Size = 1024 * 16;
    std::mt19937_64 Mt(20231125);
    std::vector<float> X(Size);
    std::vector<float> Y1(Size);
    std::vector<float> Y2(Size);

    float* DeviceIn;
    float* DeviceOut;

    cudaMalloc(&DeviceIn, Size * sizeof(float));
    cudaMalloc(&DeviceOut, Size * sizeof(float));

    std::uniform_real_distribution<float> Dist(-100.0f, 100.0f);

    for (std::size_t T = 0; T < N; ++T) {
        for (std::size_t I = 0; I < Size; ++I) {
            X[I] = Dist(Mt);
        }

        cudaMemcpy(DeviceIn, X.data(), Size * sizeof(float), cudaMemcpyHostToDevice);

        nshogi::engine::cuda::sigmoid(DeviceOut, DeviceIn, Size);

        cudaMemcpy(Y1.data(), DeviceOut, Size * sizeof(float), cudaMemcpyDeviceToHost);

        std::transform(X.begin(), X.end(), Y2.begin(), sigmoid);

        for (std::size_t I = 0; I < Size; ++I) {
            CU_ASSERT_DOUBLE_EQUAL(Y1[I], Y2[I], 1e-6);
        }
    }

    cudaFree(DeviceOut);
    cudaFree(DeviceIn);
}

} // namespace


int setupCudaMath() {
    CU_pSuite suite = CU_add_suite("cuda math test", 0, 0);

    CU_add_test(suite, "testSigmoid", testSigmoid);

    return CUE_SUCCESS;
}
