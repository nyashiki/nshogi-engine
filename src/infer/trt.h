#ifndef NSHOGI_ENGINE_INFER_TRT_H
#define NSHOGI_ENGINE_INFER_TRT_H

#include "infer.h"

#include <cstdint>
#include <string>
#include <iostream>
#include <memory>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda.h"
#include "cuda_runtime.h"


#include <thread>
#include <stdexcept>

namespace nshogi {
namespace engine {
namespace infer {

class TRTLogger : public nvinfer1::ILogger {
 public:
    void log(Severity Sev, const char* Msg) throw() override {
        if ((Sev == Severity::kERROR || Sev == Severity::kINTERNAL_ERROR)) {
            std::cerr << Msg << std::endl;
            exit(1);
        }
    }
};

class TensorRT : public Infer {
 public:
    TensorRT(int GPUId, uint16_t BatchSize, uint16_t NumChannels);
    ~TensorRT() override;

    void load(const std::string& Path, bool UseSerializedFileIfAvailable);

    void computeNonBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void computeBlocking(const ml::FeatureBitboard* Features, std::size_t BatchSize, float* DstPolicy, float* DstWinRate, float* DstDrawRate) override;
    void await() override;
    bool isComputing() override;
    void resetGPU();

 private:
    void dummyInference(std::size_t Repeat);

    const uint16_t BatchSizeM;
    const uint16_t NumC;
    const int GPUId_;

    TRTLogger Logger;
    std::unique_ptr<nvinfer1::IBuilder> Builder;
    std::unique_ptr<nvinfer1::IBuilderConfig> BuilderConfig;
    std::unique_ptr<nvinfer1::INetworkDefinition> Network;
    std::unique_ptr<nvonnxparser::IParser> Parser;
    std::unique_ptr<nvinfer1::ICudaEngine> CudaEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> Context;
    std::unique_ptr<nvinfer1::IHostMemory> Plan;
    std::unique_ptr<nvinfer1::IRuntime> Runtime;
    cudaStream_t Stream;

    void* DeviceInput;
    void* DeviceInputExtracted;
    void* DevicePolicyOutput;
    void* DeviceValueOutput;
    void* DeviceDrawOutput;
    bool Called = false;
    std::thread::id ThreadIDPrev;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_TRT_H
