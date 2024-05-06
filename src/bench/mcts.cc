#include "mcts.h"

#include "../globalconfig.h"
#include "../limit.h"
#include "../allocator/allocator.h"
#include "../mcts/manager.h"
#include "../protocol/usilogger.h"

#include <chrono>
#include <memory>

#include <nshogi/core/state.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace bench {

void benchMCTS(uint64_t DurationSeconds, std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumThreadsPerGPU, std::size_t NumCheckmateSearchers, std::size_t EvalCacheMB) {
    // Setup peripherals.
    const std::size_t AvailableMemoryGB = 8UL;
    std::cout << "Warning: this method consumes " << AvailableMemoryGB << " GB memory." << std::endl;

    allocator::getNodeAllocator().resize((std::size_t)(0.1 * (double)AvailableMemoryGB * 1024 * 1024 * 1024));
    allocator::getEdgeAllocator().resize((std::size_t)(0.9 * (double)AvailableMemoryGB * 1024 * 1024 * 1024));
    GlobalConfig::getConfig().setEvalCacheMemoryMB(EvalCacheMB);
    protocol::usi::USILogger Logger;

    // Setup MCTS.
    std::cout << "Setting MCTS." << std::endl;
    std::cout << "    - NumGPUs: " << NumGPUs << std::endl;
    std::cout << "    - NumThreadsPerGPU: " << NumThreadsPerGPU << std::endl;
    std::cout << "    - NumCheckmateSearchers: " << NumCheckmateSearchers << std::endl;
    std::cout << "    - BatchSize: " << BatchSize << std::endl;
    mcts::Manager Manager(NumGPUs, NumThreadsPerGPU, NumCheckmateSearchers, BatchSize, &Logger);

    // Setup a state.
    std::cout << "Setting a state." << std::endl;
    auto State = core::StateBuilder::getInitialState();
    auto Config = core::StateConfig();
    std::cout << "    - State: " << nshogi::io::sfen::stateToSfen(State) << std::endl;

    // Start thinking (without blocking).
    std::cout << "MCTS has started for " << DurationSeconds << " seconds." << std::endl;
    Manager.thinkNextMove(State, Config, NoLimit, nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(DurationSeconds * 1000));
    Manager.stop();

    std::cout << "MCTS done." << std::endl;
}

} // namespace bench
} // namespace engine
} // namespace nshogi
