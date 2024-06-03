#ifndef NSHOGI_ENGINE_GLOBALCONFIG_H
#define NSHOGI_ENGINE_GLOBALCONFIG_H

#include <cstdint>
#include <string>
#include <typeinfo>

#include "book/strategy.h"
#include "evaluate/preset.h"

namespace nshogi {
namespace engine {


struct GlobalConfigImpl {
 public:
    bool getPonderingEnabled() const {
        return PonderingEnabled;
    }

    std::size_t getNumGPUs() const {
        return NumGPUs;
    }

    std::size_t getNumSearchThreads() const {
        return NumSearchThreads;
    }

    std::size_t getNumEvaluationThreadsPerGPU() const {
        return NumEvaluationThreadsPerGPU;
    }

    std::size_t getNumCheckmateSearchThreads() const {
        return NumCheckmateSearchThreads;
    }

    std::size_t getBatchSize() const {
        return BatchSize;
    }

    uint32_t getThinkingTimeMargin() const {
        return ThinkingTimeMargin;
    }

    uint32_t getLogMargin() const {
        return LogMargin;
    }

    const std::string& getWeightPath() const {
        return WeightPath;
    }

    bool isBookEnabled() const {
        return IsBookEnabled;
    }

    const std::string& getBookPath() const {
        return Bookpath;
    }

    std::size_t getAvailableMemoryMB() const {
        return AvailableMemoryMB;
    }

    std::size_t getEvalCacheMemoryMB() const {
        return EvalCacheMemoryMB;
    }

    double getMemoryLimitFactor() const {
        return MemoryLimitFactor;
    }

    std::size_t getNumGarbageCollectorThreads() const {
        return NumGarbageCollectorThreads;
    }

    float getBlackDrawValue() const {
        return BlackDrawValue;
    }

    float getWhiteDrawValue() const {
        return WhiteDrawValue;
    }

    bool isRepetitionBookAllowed() const {
        return IsRepetitionBookAllowed;
    }

    book::Strategy getBookSelectionStrategy() const {
        return BookSelectionStrategy;
    }

    void setPonderingEnabled(bool Value) {
        PonderingEnabled = Value;
    }

    void setNumGPUs(std::size_t GPUs) {
        NumGPUs = GPUs;
    }

    void setNumSearchThreads(std::size_t NumThreads) {
        NumSearchThreads = NumThreads;
    }

    void setNumEvaluationThreadsPerGPU(std::size_t NumThreads) {
        NumEvaluationThreadsPerGPU = NumThreads;
    }

    void setNumCheckmateSearchThreads(std::size_t NumCheckmateThreads) {
        NumCheckmateSearchThreads = NumCheckmateThreads;
    }

    void setBatchSize(std::size_t Size) {
        BatchSize = Size;
    }

    void setThinkingTimeMargin(uint32_t Margin) {
        ThinkingTimeMargin = Margin;
    }

    void setAvailableMemoryMB(std::size_t Memory) {
        AvailableMemoryMB = Memory;
    }

    void setEvalCacheMemoryMB(std::size_t Memory) {
        EvalCacheMemoryMB = Memory;
    }

    void setMemoryLimitFactor(double Factor) {
        MemoryLimitFactor = Factor;
    }

    void setNumGarbageCollectorThreads(std::size_t Threads) {
        NumGarbageCollectorThreads = Threads;
    }

    void setWeightPath(const std::string& Path) {
        WeightPath = Path;
    }

    void setIsBookEnabled(bool Value) {
        IsBookEnabled = Value;
    }

    void setBlackDrawValue(float DrawValue) {
        BlackDrawValue = DrawValue;
    }

    void setWhiteDrawValue(float DrawValue) {
        WhiteDrawValue = DrawValue;
    }

    void setIsRepetitionBookAllowed(bool Value) {
        IsRepetitionBookAllowed = Value;
    }

    void setBookPath(const std::string& Path) {
        Bookpath = Path;
    }

    void setBookSelectionStrategy(book::Strategy St) {
        BookSelectionStrategy = St;
    }

 private:
    GlobalConfigImpl() {
    }

    // Configurable variables.
    bool PonderingEnabled = true;

    uint32_t MinimumThinkingTimeMilliSeconds = 0;
    uint32_t MaximumThinkingTimeMilliSeconds = 60 * 60 * 1000;  // one hour.

    std::size_t NumGPUs = 1;
    std::size_t NumSearchThreads = 2;
    std::size_t NumEvaluationThreadsPerGPU = 2;
    std::size_t NumCheckmateSearchThreads = 2;

    std::size_t BatchSize = 128;

    uint32_t ThinkingTimeMargin = 500;

    uint32_t LogMargin = 800;

    std::size_t AvailableMemoryMB = 8 * 1024;
    double MemoryLimitFactor = 0.7;
    std::size_t NumGarbageCollectorThreads = 2;

    std::size_t EvalCacheMemoryMB = 8 * 1024;

    float BlackDrawValue = 0.5f;
    float WhiteDrawValue = 0.5f;

    bool IsRepetitionBookAllowed = true;

    std::string WeightPath = "./res/model.onnx";

    bool IsBookEnabled = false;
    std::string Bookpath = "";

    book::Strategy BookSelectionStrategy = book::Strategy::MostVisited;

 friend struct GlobalConfig;
};

struct GlobalConfig {
    static GlobalConfigImpl& getConfig() {
        static GlobalConfigImpl Singleton;
        return Singleton;
    }

    using FeatureType = evaluate::preset::CustomFeaturesV1;
    // using FeatureType = evaluate::preset::SimpleFeatures;

 private:
    GlobalConfig();
};


} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_GLOBALCONFIG_H
