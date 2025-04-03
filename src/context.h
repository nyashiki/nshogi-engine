//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_CONTEXT_H
#define NSHOGI_ENGINE_CONTEXT_H

#include <cinttypes>
#include <string>

namespace nshogi {
namespace engine {

class Context {
 public:
    bool getPonderingEnabled() const;

    uint32_t getMinimumThinkingTimeMilliseconds() const;

    uint32_t getMaximumThinkingTimeMilliseconds() const;

    std::size_t getNumGPUs() const;

    std::size_t getNumSearchThreads() const;

    std::size_t getNumEvaluationThreadsPerGPU() const;

    std::size_t getNumCheckmateSearchThreads() const;

    std::size_t getBatchSize() const;

    uint32_t getThinkingTimeMargin() const;

    uint32_t getLogMargin() const;

    const std::string& getWeightPath() const;

    bool isBookEnabled() const;

    const std::string& getBookPath() const;

    std::size_t getAvailableMemoryMB() const;

    std::size_t getEvalCacheMemoryMB() const;

    double getMemoryLimitFactor() const;

    std::size_t getNumGarbageCollectorThreads() const;

    float getBlackDrawValue() const;

    float getWhiteDrawValue() const;

    bool isRepetitionBookAllowed() const;

 private:
    bool PonderingEnabled = true;

    uint32_t MinimumThinkingTimeMilliSeconds = 0;
    uint32_t MaximumThinkingTimeMilliSeconds = 60 * 60 * 1000; // one hour.

    std::size_t NumGPUs = 1;
    std::size_t NumSearchThreads = 2;
    std::size_t NumEvaluationThreadsPerGPU = 2;
    std::size_t NumCheckmateSearchThreads = 2;

    std::size_t BatchSize = 128;

    uint32_t ThinkingTimeMargin = 500;

    uint32_t LogMargin = 800;

    std::size_t AvailableMemoryMB = 8 * 1024;
    double MemoryLimitFactor = 0.8;
    std::size_t NumGarbageCollectorThreads = 2;

    std::size_t EvalCacheMemoryMB = 8 * 1024;

    float BlackDrawValue = 0.5f;
    float WhiteDrawValue = 0.5f;

    bool IsRepetitionBookAllowed = true;

    std::string WeightPath = "./res/model.onnx";

    bool IsBookEnabled = false;
    std::string Bookpath = "";

    friend class ContextManager;
};

} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_CONTEXT_H
