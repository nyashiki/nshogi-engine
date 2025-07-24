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

    std::ifstream Ifs(Path);

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

        // Insert sigmoid activation at value and draw output.
        {
            const int NumOutputs = Network->getNbOutputs();
            std::vector<nvinfer1::ITensor*> Targets;

            for (int I = 0; I < NumOutputs; ++I) {
                nvinfer1::ITensor* Out = Network->getOutput(I);
                const std::string Name = Out->getName();

                if (Name == "value" || Name == "draw") {
                    Targets.push_back(Out);
                }
            }

            for (auto* Out: Targets) {
                const std::string Name = Out->getName();
                const std::string OldName = Name + "_logits";
                const std::string NewName = Name + "_sigmoid";

                Network->unmarkOutput(*Out);
                Out->setName(OldName.c_str());

                auto Act = Network->addActivation(*Out, nvinfer1::ActivationType::kSIGMOID);
                Act->setName(NewName.c_str());

                nvinfer1::ITensor* OutActivated = Act->getOutput(0);
                OutActivated->setName(Name.c_str());
                Network->markOutput(*OutActivated);
            }
        }

        BuilderConfig.reset(Builder->createBuilderConfig());

        BuilderConfig->setAvgTimingIterations(8);
        BuilderConfig->setBuilderOptimizationLevel(5);
        BuilderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                          64ULL << 20);

        cudaStream_t ProfileStream;
        cudaStreamCreate(&ProfileStream);
        BuilderConfig->setProfileStream(ProfileStream);

        auto Profile = Builder->createOptimizationProfile();

        Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{1, NumC, 9, 9});
        Profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{BatchSizeM, NumC, 9, 9});
        Profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{BatchSizeM, NumC, 9, 9});
        BuilderConfig->addOptimizationProfile(Profile);

        BuilderConfig->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);

        if (Builder->platformHasFastFp16()) {
            BuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        BuilderConfig->setFlag(nvinfer1::BuilderFlag::kTF32);

        BuilderConfig->setProfileStream(Stream);
        Plan.reset(Builder->buildSerializedNetwork(*Network, *BuilderConfig));

        cudaStreamSynchronize(ProfileStream);
        cudaStreamDestroy(ProfileStream);

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

    Context->setInputTensorAddress("input", DeviceInputExtracted);
    Context->setTensorAddress("policy", DevicePolicyOutput);
    Context->setTensorAddress("value", DeviceValueOutput);
    Context->setTensorAddress("draw", DeviceDrawOutput);

    makeCudaGraph();

    Called = false;
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
        Context->setInputShape("input",
                               nvinfer1::Dims4{(int32_t)(BatchSizeM + 1) / 2, NumC, 9, 9});

        cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal);

        cuda::extractBits(reinterpret_cast<float*>(DeviceInputExtracted),
                          reinterpret_cast<uint64_t*>(DeviceInput),
                          (int)(BatchSizeM + 1) / 2 * (int)NumC, Stream);

        Context->enqueueV3(Stream);

        cudaStreamEndCapture(Stream, &CudaGraph[0]);

        cudaGraphInstantiate(&CudaGraphExec[0], CudaGraph[0], nullptr, nullptr, 0);

        cudaGraphUpload(CudaGraphExec[0], Stream);

        cudaStreamSynchronize(Stream);
    }

    { // Cuda graph for full batch size.
        Context->setInputShape("input",
                               nvinfer1::Dims4{(int32_t)BatchSizeM, NumC, 9, 9});

        cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal);

        cuda::extractBits(reinterpret_cast<float*>(DeviceInputExtracted),
                          reinterpret_cast<uint64_t*>(DeviceInput),
                          (int)BatchSizeM * (int)NumC, Stream);

        Context->enqueueV3(Stream);

        cudaStreamEndCapture(Stream, &CudaGraph[1]);

        cudaGraphInstantiate(&CudaGraphExec[1], CudaGraph[1], nullptr, nullptr, 0);

        cudaGraphUpload(CudaGraphExec[1], Stream);

        cudaStreamSynchronize(Stream);
    }
}

} // namespace infer
} // namespace engine
} // namespace nshogi
