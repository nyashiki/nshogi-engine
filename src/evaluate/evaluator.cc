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

Evaluator::Evaluator([[maybe_unused]] std::size_t ThreadId, std::size_t FeatureSize, std::size_t BatchSize, infer::Infer* In)
    : PInfer(In)
    , MyFeatureSize(FeatureSize)
    , BatchSizeMax(BatchSize)
    , NumaUsed(false)
    , MyNumaId(0) {
#ifdef NUMA_ENABLED
    const int NumaAvailable = numa_available();
    if (NumaAvailable < 0) {
        std::cout << "Warning: numa is enabled by numa_available() returns " << NumaAvailable << std::endl;
    } else {
        // Fetch available NUMA nodes.
        struct bitmask *NumaNodes = numa_all_nodes_ptr;
        for (unsigned int I = 0; I < NumaNodes->size; ++I) {
            if (numa_bitmask_isbitset(NumaNodes, I)) {
                AvailableNumaNodes.push_back((int)I);
            }
        }

        // Specify my numa node.
        MyNumaId = AvailableNumaNodes[ThreadId % AvailableNumaNodes.size()];

        // Fetch thread indices associated with my numa node.
        struct bitmask *CPUMask = numa_allocate_cpumask();
        numa_node_to_cpus(MyNumaId, CPUMask);

        std::cout << "ThreadId: " << ThreadId << ", NumaId: " << MyNumaId << ", ";

        // Set affinity.
        const std::size_t CPUSetSize = CPU_ALLOC_SIZE(CPUMask->size);
        cpu_set_t *CPUSet = CPU_ALLOC(CPUMask->size);
        CPU_ZERO_S(CPUSetSize, CPUSet);

        std::cout << "CPUMask: ";
        for (unsigned int I = 0; I < CPUMask->size; ++I) {
            if (numa_bitmask_isbitset(CPUMask, I)) {
                CPU_SET_S(I, CPUSetSize, CPUSet);
                std::cout << I << ", ";
            }
        }
        std::cout << std::endl;
        sched_setaffinity(0, CPUSetSize, CPUSet);

        // Release the resources.
        CPU_FREE(CPUSet);
        numa_free_cpumask(CPUMask);

        NumaUsed = true;
    }
#endif
    FeatureBitboards = static_cast<ml::FeatureBitboard*>(allocateMemoryByNumaIfAvailable(
                BatchSizeMax * MyFeatureSize * sizeof(ml::FeatureBitboard)));
    Policy = static_cast<float*>(allocateMemoryByNumaIfAvailable(
                ml::MoveIndexMax * BatchSizeMax * sizeof(float)));
    WinRate = static_cast<float*>(allocateMemoryByNumaIfAvailable(
                BatchSizeMax * sizeof(float)));
    DrawRate = static_cast<float*>(allocateMemoryByNumaIfAvailable(
                BatchSizeMax * sizeof(float)));

#ifdef CUDA_ENABLED
    // Pin the memory.
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

    freeMemory(reinterpret_cast<void**>(&FeatureBitboards), BatchSizeMax * MyFeatureSize * sizeof(ml::FeatureBitboard));
    freeMemory(reinterpret_cast<void**>(&Policy), ml::MoveIndexMax * BatchSizeMax * sizeof(float));
    freeMemory(reinterpret_cast<void**>(&WinRate), BatchSizeMax * sizeof(float));
    freeMemory(reinterpret_cast<void**>(&DrawRate), BatchSizeMax * sizeof(float));
}

void* Evaluator::allocateMemoryByNumaIfAvailable(std::size_t Size) const {
#ifdef NUMA_ENABLED
    if (NumaUsed) {
        void* Memory = numa_alloc_onnode(Size, MyNumaId);
        return Memory;
    }
#endif
    void* Memory = std::malloc(Size);
    return Memory;
}

void Evaluator::freeMemory(void** Memory, [[maybe_unused]] std::size_t Size) const {
#ifdef NUMA_ENABLED
    if (NumaUsed) {
        numa_free(*Memory, Size);
        *Memory = nullptr;
        return;
    }
#endif
    std::free(*Memory);
    *Memory = nullptr;
}

} // namespace evaluate
} // namespace engine
} // namespace nshogi
