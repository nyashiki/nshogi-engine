//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evaluator.h"

#include <iostream>

#ifdef CUDA_ENABLED

#include <cuda_runtime.h>

#endif

#ifdef NUMA_ENABLED

#include <numa.h>
#include <numaif.h>

#endif

namespace nshogi {
namespace engine {
namespace evaluate {

Evaluator::Evaluator(std::size_t ThreadId, std::size_t FeatureSize, std::size_t BatchSize, infer::Infer* In)
    : PInfer(In)
    , MyFeatureSize(FeatureSize)
    , BatchSizeMax(BatchSize)
    , NumaUsed(false) {
#ifdef NUMA_ENABLED
    const int NumaAvailable = numa_available();
    if (NumaAvailable < 0) {
        std::cout << "Warning: numa is enabled by numa_available() returns " << NumaAvailable << std::endl;
        FeatureBitboards = new ml::FeatureBitboard[BatchSizeMax * MyFeatureSize];
        Policy = new float[ml::MoveIndexMax * BatchSizeMax];
        WinRate = new float[BatchSizeMax];
        DrawRate = new float[BatchSizeMax];
    } else {
        const int NumaMaxNode = numa_max_node();
        const int BindId = (int)ThreadId % (NumaMaxNode + 1);

        numa_run_on_node(BindId);

        FeatureBitboards = static_cast<ml::FeatureBitboard*>(numa_alloc_local(BatchSizeMax * MyFeatureSize * sizeof(ml::FeatureBitboard)));
        Policy = static_cast<float*>(numa_alloc_local(ml::MoveIndexMax * BatchSizeMax * sizeof(float)));
        WinRate = static_cast<float*>(numa_alloc_local(BatchSizeMax * sizeof(float)));
        DrawRate = static_cast<float*>(numa_alloc_local(BatchSizeMax * sizeof(float)));

        NumaUsed = true;
    }
#else
    FeatureBitboards = new ml::FeatureBitboard[BatchSizeMax * MyFeatureSize];
    Policy = new float[ml::MoveIndexMax * BatchSizeMax];
    WinRate = new float[BatchSizeMax];
    DrawRate = new float[BatchSizeMax];
#endif

#ifdef CUDA_ENABLED
    cudaHostRegister(FeatureBitboards, BatchSizeMax * MyFeatureSize * sizeof(ml::FeatureBitboard), cudaHostRegisterDefault);
    cudaHostRegister(Policy, ml::MoveIndexMax * BatchSizeMax * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(WinRate, BatchSizeMax * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(DrawRate, BatchSizeMax * sizeof(float), cudaHostRegisterDefault);
#endif
}

Evaluator::~Evaluator() {
#ifdef CUDA_ENABLED
    cudaHostUnregister(FeatureBitboards);
    cudaHostUnregister(Policy);
    cudaHostUnregister(WinRate);
    cudaHostUnregister(DrawRate);
#endif

#ifdef NUMA_ENABLED
    if (NumaUsed) {
        numa_free(FeatureBitboards, BatchSizeMax * MyFeatureSize * sizeof(ml::FeatureBitboard));
        numa_free(Policy, ml::MoveIndexMax * BatchSizeMax * sizeof(float));
        numa_free(WinRate, BatchSizeMax * sizeof(float));
        numa_free(DrawRate, BatchSizeMax * sizeof(float));
    } else {
        delete[] FeatureBitboards;
        delete[] Policy;
        delete[] WinRate;
        delete[] DrawRate;
    }
#else
    delete[] FeatureBitboards;
    delete[] Policy;
    delete[] WinRate;
    delete[] DrawRate;
#endif
}


} // namespace evaluate
} // namespace engine
} // namespace nshogi
