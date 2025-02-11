//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <gtest/gtest.h>

#include "../cuda/math.h"

#include <algorithm>
#include <random>
#include <vector>

namespace {

template <typename T>
T sigmoid(T x) {
    return (T)(1.0 / (1.0 + std::exp(-(double)x)));
}

} // namespace

TEST(CUDA, Sigmoid) {
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

        std::transform(X.begin(), X.end(), Y2.begin(), sigmoid<float>);

        for (std::size_t I = 0; I < Size; ++I) {
            EXPECT_NEAR(Y1[I], Y2[I], 1e-6);
        }
    }

    cudaFree(DeviceOut);
    cudaFree(DeviceIn);
}
