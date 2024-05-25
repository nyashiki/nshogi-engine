#ifndef NSHOGI_ENGINE_MCTS_WATCHDOG_H
#define NSHOGI_ENGINE_MCTS_WATCHDOG_H

#include "edge.h"
#include "node.h"
#include "../limit.h"
#include "../worker/worker.h"
#include "../logger/logger.h"

#include <functional>
#include <memory>
#include <vector>

#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>

namespace nshogi {
namespace engine {
namespace mcts {

class Watchdog : public worker::Worker {
 public:
    Watchdog(std::shared_ptr<logger::Logger>);
    ~Watchdog();

    void updateRoot(const core::State*, const core::StateConfig*, Node*);
    void setLimit(const engine::Limit&);
    void setStopSearchingCallback(std::function<void()> Callback);

 private:
    bool doTask() override;

    bool isRootSolved() const;
    bool checkMemoryBudget() const;
    bool checkThinkingTimeBudget(uint32_t) const;
    bool hasMadeUpMind(uint32_t);

    void dumpPVLog(uint64_t, uint32_t) const;
    logger::PVLog getPVLog() const;

    const core::State* State;
    const core::StateConfig* Config;
    Node* Root;
    std::unique_ptr<engine::Limit> Limit;

    Edge* BestEdgePrevious;
    std::vector<double> VisitsPrevious;
    uint32_t ElapsedPrevious;

    std::function<void()> StopSearchingCallback;

    std::shared_ptr<logger::Logger> PLogger;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_WATCHDOG_H
