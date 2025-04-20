//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_WATCHDOG_H
#define NSHOGI_ENGINE_MCTS_WATCHDOG_H

#include "../allocator/allocator.h"
#include "../context.h"
#include "../limit.h"
#include "../logger/logger.h"
#include "../worker/worker.h"
#include "edge.h"
#include "node.h"

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
    Watchdog(const Context*, allocator::Allocator* NodeAllocator,
             allocator::Allocator* EdgeAllocator,
             std::shared_ptr<logger::Logger>);
    ~Watchdog();

    void updateRoot(const core::State*, const core::StateConfig*, Node*);
    void setLimit(const engine::Limit&);
    void setStopSearchingCallback(std::function<void()> Callback);

 private:
    bool doTask() override;

    bool isRootSolved() const;
    bool checkNodeLimit() const;
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

    const Context* PContext;
    allocator::Allocator* NA;
    allocator::Allocator* EA;
    std::shared_ptr<logger::Logger> PLogger;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_WATCHDOG_H
