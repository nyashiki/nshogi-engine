//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "trt.h"
#include "../cuda/extractbit.h"
#include "../globalconfig.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <sstream>

#include <nshogi/ml/common.h>
#include <nshogi/ml/featurebitboard.h>

namespace nshogi {
namespace engine {
namespace infer {

namespace {

uint64_t computeFileHash(const std::string& Path) {
    const uint64_t FNVOffsetBasis = 14695981039346656037ULL;
    const uint64_t FNVPrime = 1099511628211ULL;

    std::ifstream Ifs(Path, std::ios::binary);

    uint64_t HashValue = FNVOffsetBasis;

    while (!Ifs.eof()) {
        char C;
        Ifs.read(&C, sizeof(char));

        HashValue = (FNVPrime * HashValue) ^ (uint64_t)C;
    }

    return HashValue;
}

} // namespace

TensorRT::TensorRT(int GPUId, uint16_t BatchSizeMax, uint16_t NumChannels)
    : BatchSizeM(BatchSizeMax)
    , NumC(NumChannels)
    , GPUId_(GPUId) {
    cudaSetDevice(GPUId_);
    cudaMalloc(reinterpret_cast<void**>(&DeviceInput),
               BatchSizeMax * NumChannels * sizeof(ml::FeatureBitboard));
    cudaMalloc(reinterpret_cast<void**>(&DeviceInputExtracted),
               BatchSizeMax * NumChannels * core::NumSquares * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&DevicePolicyOutput),
               BatchSizeMax * ml::MoveIndexMax * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&DeviceValueOutput),
               BatchSizeMax * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&DeviceDrawOutput),
               BatchSizeMax * sizeof(float));

    cudaMemset(reinterpret_cast<void*>(DeviceInput), 0,
               BatchSizeMax * NumChannels * sizeof(ml::FeatureBitboard));
    cudaMemset(reinterpret_cast<void*>(DeviceInputExtracted), 0,
               BatchSizeMax * NumChannels * core::NumSquares * sizeof(float));
    cudaMemset(reinterpret_cast<void*>(DevicePolicyOutput), 0,
               BatchSizeMax * ml::MoveIndexMax * sizeof(float));
    cudaMemset(reinterpret_cast<void*>(DeviceValueOutput), 0,
               BatchSizeMax * sizeof(float));
    cudaMemset(reinterpret_cast<void*>(DeviceDrawOutput), 0,
               BatchSizeMax * sizeof(float));

    cudaStreamCreateWithFlags(&Stream, cudaStreamNonBlocking);
}

TensorRT::~TensorRT() {
    Context.reset();
    CudaEngine.reset();

    cudaFree(DeviceInput);
    cudaFree(DeviceInputExtracted);
    cudaFree(DevicePolicyOutput);
    cudaFree(DeviceValueOutput);
    cudaFree(DeviceDrawOutput);

    cudaGraphExecDestroy(CudaGraphExec[0]);
    cudaGraphDestroy(CudaGraph[0]);
    cudaGraphExecDestroy(CudaGraphExec[1]);
    cudaGraphDestroy(CudaGraph[1]);
    cudaStreamDestroy(Stream);

    Plan.reset();
    Parser.reset();
    Network.reset();
    BuilderConfig.reset();
    Builder.reset();
    Runtime.reset();
}

void TensorRT::load(const std::string& Path,
                    bool UseSerializedFileIfAvailable) {
    const std::string SerializedPath = [&Path]() {
        std::stringstream SS;
        SS << std::hex << computeFileHash(Path) << ".serialized";
        return SS.str();
    }();

    std::ifstream SerializedIfs(SerializedPath, std::ios::binary);

    if (!UseSerializedFileIfAvailable || !SerializedIfs.is_open()) {
        Builder.reset(nvinfer1::createInferBuilder(Logger));
        // Network.reset(Builder->createNetworkV2(1U <<
        // static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        Network.reset(Builder->createNetworkV2(0));

        Parser.reset(nvonnxparser::createParser(*Network, Logger));

        if (!Parser->parseFromFile(
                Path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
            throw std::runtime_error("Could not parse the model.");
        }

        BuilderConfig.reset(Builder->createBuilderConfig());

        BuilderConfig->setAvgTimingIterations(8);
        BuilderConfig->setBuilderOptimizationLevel(5);
        BuilderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                          64ULL << 20);

        BuilderConfig->setProfileStream(Stream);

        auto Profile = Builder->createOptimizationProfile();

        if constexpr (global_config::ChannelsFirst) {
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4{1, NumC, 9, 9});
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4{BatchSizeM, NumC, 9, 9});
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4{BatchSizeM, NumC, 9, 9});
        } else {
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4{1, 9, 9, NumC});
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4{BatchSizeM, 9, 9, NumC});
            Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4{BatchSizeM, 9, 9, NumC});
        }
        BuilderConfig->addOptimizationProfile(Profile);

        BuilderConfig->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);

        if (Builder->platformHasFastFp16()) {
            BuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        BuilderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);

        Plan.reset(Builder->buildSerializedNetwork(*Network, *BuilderConfig));

        cudaStreamSynchronize(Stream);

        Runtime.reset(nvinfer1::createInferRuntime(Logger));
        CudaEngine.reset(
            Runtime->deserializeCudaEngine(Plan->data(), Plan->size()));

        // Save the serialized file.
        std::unique_ptr<nvinfer1::IHostMemory> Serialized{
            CudaEngine->serialize()};
        std::ofstream SerializedOfs(SerializedPath, std::ios::binary);
        SerializedOfs.write(reinterpret_cast<char*>(Serialized->data()),
                            (std::streamsize)Serialized->size());
    } else {
        SerializedIfs.seekg(0, std::ios_base::end);
        const std::size_t FileSize = (std::size_t)SerializedIfs.tellg();
        SerializedIfs.seekg(0, std::ios_base::beg);

        std::vector<char> Blob(FileSize);
        SerializedIfs.read(Blob.data(), (std::streamsize)FileSize);
        Runtime.reset(nvinfer1::createInferRuntime(Logger));
        CudaEngine.reset(Runtime->deserializeCudaEngine(Blob.data(), FileSize));
    }

    Context.reset(CudaEngine->createExecutionContext());
    Context->setOptimizationProfileAsync(0, Stream);

    // Check policy size.
    const nvinfer1::Dims PolicyDims = Context->getTensorShape("policy");
    const std::size_t PolicySize = [PolicyDims]() -> std::size_t {
        std::size_t Size = 1;
        for (std::size_t I = 0; I < (std::size_t)PolicyDims.nbDims; ++I) {
            if (PolicyDims.d[I] == -1) {
                continue;
            }
            Size *= (std::size_t)PolicyDims.d[I];
        }
        return Size;
    }();

    if (PolicySize != nshogi::ml::MoveIndexMax) {
        std::cerr << "Unexpected PolicySize: " << PolicySize
                  << " (expected: " << nshogi::ml::MoveIndexMax << ")."
                  << std::endl;
        std::abort();
    }

    Context->setInputTensorAddress("input", DeviceInputExtracted);
    if (!Context->setTensorAddress("policy", DevicePolicyOutput)) {
        std::cerr << "[load()] Context->setTensorAddress() for policy failed." << std::endl;
        std::abort();
    }
    if (!Context->setTensorAddress("value", DeviceValueOutput)) {
        std::cerr << "[load()] Context->setTensorAddress() for value failed." << std::endl;
        std::abort();
    }
    if (!Context->setTensorAddress("draw", DeviceDrawOutput)) {
        std::cerr << "[load()] Context->setTensorAddress() for draw failed." << std::endl;
        std::abort();
    }

    makeCudaGraph();
}

void TensorRT::computeNonBlocking(const ml::FeatureBitboard* Features,
                                  std::size_t BatchSize, float* DstPolicy,
                                  float* DstWinRate, float* DstDrawRate) {
    assert(BatchSize <= BatchSizeM);
    assert(!isComputing());

    cudaMemcpyAsync(DeviceInput, Features,
                    BatchSize * NumC * sizeof(ml::FeatureBitboard),
                    cudaMemcpyHostToDevice, Stream);

    cudaGraphLaunch(CudaGraphExec[BatchSize > (BatchSizeM + 1) / 2], Stream);

    // Copy GPU output onto CPU.
    cudaMemcpyAsync(DstPolicy, DevicePolicyOutput,
                    BatchSize * nshogi::ml::MoveIndexMax * sizeof(float),
                    cudaMemcpyDeviceToHost, Stream);
    cudaMemcpyAsync(DstWinRate, DeviceValueOutput, BatchSize * sizeof(float),
                    cudaMemcpyDeviceToHost, Stream);
    cudaMemcpyAsync(DstDrawRate, DeviceDrawOutput, BatchSize * sizeof(float),
                    cudaMemcpyDeviceToHost, Stream);
}

void TensorRT::computeBlocking(const ml::FeatureBitboard* Features,
                               std::size_t BatchSize, float* DstPolicy,
                               float* DstWinRate, float* DstDrawRate) {
    computeNonBlocking(Features, BatchSize, DstPolicy, DstWinRate, DstDrawRate);
    await();
}

void TensorRT::await() {
    cudaStreamSynchronize(Stream);
}

bool TensorRT::isComputing() {
    return cudaStreamQuery(Stream) == cudaErrorNotReady;
}

void TensorRT::resetGPU() {
    cudaSetDevice(GPUId_);
}

void TensorRT::makeCudaGraph() {
    { // Cuda graph for half batch size.
        if constexpr (global_config::ChannelsFirst) {
            if (!Context->setInputShape(
                    "input", nvinfer1::Dims4{(int32_t)(BatchSizeM + 1) / 2,
                                             NumC, 9, 9})) {
                std::cerr
                    << "[makeCudaGraph()] Context->setInputShape() failed."
                    << std::endl;
                std::abort();
            }
        } else {
            if (!Context->setInputShape(
                    "input", nvinfer1::Dims4{(int32_t)(BatchSizeM + 1) / 2, 9,
                                             9, NumC})) {
                std::cerr
                    << "[makeCudaGraph()] Context->setInputShape() failed."
                    << std::endl;
                std::abort();
            }
        }

        cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal);

        cuda::extractBits<global_config::ChannelsFirst>(
            reinterpret_cast<float*>(DeviceInputExtracted),
            reinterpret_cast<uint64_t*>(DeviceInput), (int)(BatchSizeM + 1) / 2,
            (int)NumC, Stream);

        Context->enqueueV3(Stream);

        cudaStreamEndCapture(Stream, &CudaGraph[0]);

        cudaGraphInstantiate(&CudaGraphExec[0], CudaGraph[0], nullptr, nullptr,
                             0);

        cudaGraphUpload(CudaGraphExec[0], Stream);

        cudaStreamSynchronize(Stream);
    }

    { // Cuda graph for full batch size.
        if constexpr (global_config::ChannelsFirst) {
            if (!Context->setInputShape(
                    "input",
                    nvinfer1::Dims4{(int32_t)BatchSizeM, NumC, 9, 9})) {
                std::cerr
                    << "[makeCudaGraph()] Context->setInputShape() failed."
                    << std::endl;
                std::abort();
            }
        } else {
            if (!Context->setInputShape(
                    "input",
                    nvinfer1::Dims4{(int32_t)BatchSizeM, 9, 9, NumC})) {
                std::cerr
                    << "[makeCudaGraph()] Context->setInputShape() failed."
                    << std::endl;
                std::abort();
            }
        }

        cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal);

        cuda::extractBits<global_config::ChannelsFirst>(
            reinterpret_cast<float*>(DeviceInputExtracted),
            reinterpret_cast<uint64_t*>(DeviceInput), (int)BatchSizeM,
            (int)NumC, Stream);

        Context->enqueueV3(Stream);

        cudaStreamEndCapture(Stream, &CudaGraph[1]);

        cudaGraphInstantiate(&CudaGraphExec[1], CudaGraph[1], nullptr, nullptr,
                             0);

        cudaGraphUpload(CudaGraphExec[1], Stream);

        cudaStreamSynchronize(Stream);
    }
}

} // namespace infer
} // namespace engine
} // namespace nshogi
