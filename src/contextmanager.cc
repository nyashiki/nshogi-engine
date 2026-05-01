//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "contextmanager.h"

namespace nshogi {
namespace engine {

ContextManager::ContextManager()
    : Context_(std::make_unique<Context>()) {
}

const Context* ContextManager::getContext() const {
    return Context_.get();
}

void ContextManager::setIsPonderingEnabled(bool Value) {
    Context_->IsPonderingEnabled = Value;
}

void ContextManager::setMinimumThinkinTimeMilliSeconds(uint32_t Value) {
    Context_->MinimumThinkingTimeMilliSeconds = Value;
}

void ContextManager::setMaximumThinkinTimeMilliSeconds(uint32_t Value) {
    Context_->MaximumThinkingTimeMilliSeconds = Value;
}

void ContextManager::setNumGPUs(std::size_t GPUs) {
    Context_->NumGPUs = GPUs;
}

void ContextManager::setNumSearchThreads(std::size_t NumThreads) {
    Context_->NumSearchThreads = NumThreads;
}

void ContextManager::setNumEvaluationThreadsPerGPU(std::size_t NumThreads) {
    Context_->NumEvaluationThreadsPerGPU = NumThreads;
}

void ContextManager::setNumFeedThreads(std::size_t NumFeedThreads) {
    Context_->NumFeedThreads = NumFeedThreads;
}

void ContextManager::setBatchSize(std::size_t Size) {
    Context_->BatchSize = Size;
}

void ContextManager::setThinkingTimeMargin(uint32_t Margin) {
    Context_->ThinkingTimeMargin = Margin;
}

void ContextManager::setAvailableMemoryMB(std::size_t Memory) {
    Context_->AvailableMemoryMB = Memory;
}

void ContextManager::setEvalCacheMemoryMB(std::size_t Memory) {
    Context_->EvalCacheMemoryMB = Memory;
}

void ContextManager::setMemoryLimitFactor(double Factor) {
    Context_->MemoryLimitFactor = Factor;
}

void ContextManager::setNumGarbageCollectorThreads(std::size_t Threads) {
    Context_->NumGarbageCollectorThreads = Threads;
}

void ContextManager::setWeightPath(const std::string& Path) {
    Context_->WeightPath = Path;
}

void ContextManager::setBookEnabled(bool Value) {
    Context_->IsBookEnabled = Value;
}

void ContextManager::setRepetitionBookAllowed(bool Value) {
    Context_->IsRepetitionBookAllowed = Value;
}

void ContextManager::setBookPath(const std::string& Path) {
    Context_->Bookpath = Path;
}

void ContextManager::setIsNShogiExtensionLogEnabled(bool Value) {
    Context_->IsNShogiExtensionLogEnabled = Value;
}

void ContextManager::setPrintStatistics(bool Value) {
    Context_->PrintStatistics = Value;
}

} // namespace engine
} // namespace nshogi
