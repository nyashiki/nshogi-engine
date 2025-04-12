//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "onnxruntime.h"

#include <math.h>

///
/// @fn sigmoid
/// @brief Do sigmoid computation.
///
static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

onnxruntime_t* createOnnxRuntimeInstance(const char* model_path) {
    onnxruntime_t* onnxruntime = (onnxruntime_t*)malloc(sizeof(onnxruntime_t));

    onnxruntime->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    onnxruntime->env = NULL;
    onnxruntime->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime", &onnxruntime->env);

    onnxruntime->session_options = NULL;
    onnxruntime->ort->CreateSessionOptions(&onnxruntime->session_options);
    OrtCUDAProviderOptions options;
    options.device_id = 0;
    options.arena_extend_strategy = 0;
    options.gpu_mem_limit = 1L * 1024 * 1024 * 1024;
    options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    options.do_copy_in_default_stream = 1;
    options.user_compute_stream = NULL;
    options.default_memory_arena_cfg = NULL;
    onnxruntime->ort->SessionOptionsAppendExecutionProvider_CUDA(onnxruntime->session_options, &options);
    onnxruntime->ort->SetIntraOpNumThreads(onnxruntime->session_options, 1);
    onnxruntime->ort->SetSessionGraphOptimizationLevel(onnxruntime->session_options, ORT_ENABLE_ALL);

    onnxruntime->ort->CreateSession(
            onnxruntime->env,
            model_path,
            onnxruntime->session_options,
            &onnxruntime->session);

    onnxruntime->memory_info = NULL;
    onnxruntime->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &onnxruntime->memory_info);

    onnxruntime->input = (float*)malloc(1 * 93 * 9 * 9 * sizeof(float));

    memset(onnxruntime->input, 0, 1 * 93 * 9 * 9 * sizeof(float));

    onnxruntime->policy = (float*)malloc(1 * 27 * 9 * 9 * sizeof(float));
    onnxruntime->value = (float*)malloc(1 * sizeof(float));
    onnxruntime->draw = (float*)malloc(1 * sizeof(float));

    onnxruntime->input_tensor = NULL;
    const int64_t dims[4] = {1, 93, 9, 9};
    size_t num_dims = sizeof(dims) / sizeof(dims[0]);
    onnxruntime->ort->CreateTensorWithDataAsOrtValue(
            onnxruntime->memory_info,
            onnxruntime->input,
            1 * 93 * 9 * 9 * sizeof(float),
            dims,
            num_dims,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &onnxruntime->input_tensor);

    return onnxruntime;
}

void destroyOnnxRuntimeInstance(onnxruntime_t* onnxruntime) {
    onnxruntime->ort->ReleaseValue(onnxruntime->input_tensor);
    onnxruntime->ort->ReleaseMemoryInfo(onnxruntime->memory_info);
    onnxruntime->ort->ReleaseSession(onnxruntime->session);
    onnxruntime->ort->ReleaseSessionOptions(onnxruntime->session_options);
    onnxruntime->ort->ReleaseEnv(onnxruntime->env);

    free(onnxruntime->input);
    free(onnxruntime->policy);
    free(onnxruntime->value);

    free(onnxruntime);
}

void runSession(onnxruntime_t* onnxruntime) {
    OrtValue* output_tensors[3] = {NULL, NULL, NULL};

    const char* input_names[1] = {"input"};
    const char* output_names[3] = {"policy", "value", "draw"};

    onnxruntime->ort->Run(
            onnxruntime->session,
            NULL,
            input_names,
            &onnxruntime->input_tensor,
            1,
            output_names,
            3,
            output_tensors);

    float* policy;
    float* value;
    float* draw;
    onnxruntime->ort->GetTensorMutableData(output_tensors[0], (void**)&policy);
    onnxruntime->ort->GetTensorMutableData(output_tensors[1], (void**)&value);
    onnxruntime->ort->GetTensorMutableData(output_tensors[2], (void**)&draw);

    memcpy(onnxruntime->policy, policy, 1 * 27 * 9 * 9 * sizeof(float));
    onnxruntime->value[0] = sigmoid(value[0]);
    onnxruntime->draw[0] = sigmoid(draw[0]);

    onnxruntime->ort->ReleaseValue(output_tensors[0]);
    onnxruntime->ort->ReleaseValue(output_tensors[1]);
    onnxruntime->ort->ReleaseValue(output_tensors[2]);
}
