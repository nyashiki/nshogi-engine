#ifndef NSHOGI_ENGINE_CONTEXTMANAGER_H
#define NSHOGI_ENGINE_CONTEXTMANAGER_H

#include "context.h"

#include <memory>

namespace nshogi {
namespace engine {

class ContextManager {
 public:
    const Context* getContext() const;

    void setPonderingEnabled(bool Value);

    void setNumGPUs(std::size_t GPUs);

    void setNumSearchThreads(std::size_t NumThreads);

    void setNumEvaluationThreadsPerGPU(std::size_t NumThreads);

    void setNumCheckmateSearchThreads(std::size_t NumCheckmateThreads);

    void setBatchSize(std::size_t Size);

    void setThinkingTimeMargin(uint32_t Margin);

    void setAvailableMemoryMB(std::size_t Memory);

    void setEvalCacheMemoryMB(std::size_t Memory);

    void setMemoryLimitFactor(double Factor);

    void setNumGarbageCollectorThreads(std::size_t Threads);

    void setWeightPath(const std::string& Path);

    void setIsBookEnabled(bool Value);

    void setBlackDrawValue(float DrawValue);

    void setWhiteDrawValue(float DrawValue);

    void setIsRepetitionBookAllowed(bool Value);

    void setBookPath(const std::string& Path);

 private:
    std::unique_ptr<Context> Context_;

};

} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_CONTEXTMANAGER_H
