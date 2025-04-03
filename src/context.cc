//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "context.h"

namespace nshogi {
namespace engine {

bool Context::getPonderingEnabled() const {
    return PonderingEnabled;
}

uint32_t Context::getMinimumThinkingTimeMilliseconds() const {
    return MinimumThinkingTimeMilliSeconds;
}

uint32_t Context::getMaximumThinkingTimeMilliseconds() const {
    return MaximumThinkingTimeMilliSeconds;
}

std::size_t Context::getNumGPUs() const {
    return NumGPUs;
}

std::size_t Context::getNumSearchThreads() const {
    return NumSearchThreads;
}

std::size_t Context::getNumEvaluationThreadsPerGPU() const {
    return NumEvaluationThreadsPerGPU;
}

std::size_t Context::getNumCheckmateSearchThreads() const {
    return NumCheckmateSearchThreads;
}

std::size_t Context::getBatchSize() const {
    return BatchSize;
}

uint32_t Context::getThinkingTimeMargin() const {
    return ThinkingTimeMargin;
}

uint32_t Context::getLogMargin() const {
    return LogMargin;
}

const std::string& Context::getWeightPath() const {
    return WeightPath;
}

bool Context::isBookEnabled() const {
    return IsBookEnabled;
}

const std::string& Context::getBookPath() const {
    return Bookpath;
}

std::size_t Context::getAvailableMemoryMB() const {
    return AvailableMemoryMB;
}

std::size_t Context::getEvalCacheMemoryMB() const {
    return EvalCacheMemoryMB;
}

double Context::getMemoryLimitFactor() const {
    return MemoryLimitFactor;
}

std::size_t Context::getNumGarbageCollectorThreads() const {
    return NumGarbageCollectorThreads;
}

float Context::getBlackDrawValue() const {
    return BlackDrawValue;
}

float Context::getWhiteDrawValue() const {
    return WhiteDrawValue;
}

bool Context::isRepetitionBookAllowed() const {
    return IsRepetitionBookAllowed;
}

} // namespace engine
} // namespace nshogi
