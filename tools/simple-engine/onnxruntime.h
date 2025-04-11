//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef ONNXRUNTIME_H
#define ONNXRUNTIME_H

#include <onnxruntime_c_api.h>

typedef struct {
    const OrtApi* ort;
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    OrtValue* input_tensor;

    float* input;
    float* policy;
    float* value;
    float* draw;
} onnxruntime_t;

onnxruntime_t* createOnnxRuntimeInstance(const char* model_path);
void destroyOnnxRuntimeInstance(onnxruntime_t* onnx_runtime);

void runSession(onnxruntime_t* onnxruntime);

#endif // #ifndef ONNXRUNTIME_H
