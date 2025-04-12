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

///
/// @struct onnxruntime_t
/// @brief Struct for neural network computation by onnxruntime.
///
typedef struct {
    /// @brief OnnxRuntime API.
    const OrtApi* ort;

    /// @brief OnnxRuntime environment.
    OrtEnv* env;

    /// @brief OnnxRuntime session options.
    OrtSessionOptions* session_options;

    /// @brief OnnxRuntime session.
    OrtSession* session;

    /// @brief OnnxRuntime memory info.
    OrtMemoryInfo* memory_info;

    /// @brief OnnxRuntine input tensor.
    OrtValue* input_tensor;

    /// @brief Input for the neural network computation.
    float* input;

    /// @brief Output for the policy.
    float* policy;

    /// @brief Output for the value.
    float* value;

    /// @brief Outpu for the draw.
    float* draw;
} onnxruntime_t;

///
/// @fn createOnnxRuntimeInstance
/// @brief Create onnxruntime_t instance.
///        This instance must be destroyed by
///        `destroyOnnxRuntimeInstance()`.
/// @param model_path The path to the onnx file.
/// @return onnxruntime_t instance.
///
onnxruntime_t* createOnnxRuntimeInstance(const char* model_path);

///
/// @fn destroyOnnxRuntimeInstance
/// @brief Destroy onnxruntime_t instance.
/// @pram onnx_runtime Instance to destroy.
///
void destroyOnnxRuntimeInstance(onnxruntime_t* onnx_runtime);

///
/// @fn runSession
/// Do inference.
/// The input must be set on onnxruntime_t->input.
/// The result is stored on onnxruntime_t->policy,
/// onnxruntime_t->value, and onnxruntime_t->draw.
/// @brief Do inference for the given input.
/// @param onnxruntime_t instnace.
///
void runSession(onnxruntime_t* onnxruntime);

#endif // #ifndef ONNXRUNTIME_H
