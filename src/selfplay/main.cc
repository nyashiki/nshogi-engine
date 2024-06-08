#include "framequeue.h"
#include "worker.h"
#include "../allocator/allocator.h"

int main() {
    constexpr std::size_t AVAILABLE_MEMORY = 4 * 1024ULL;
    constexpr std::size_t NUM_SEARCH_WORKERS = 1;
    constexpr std::size_t NUM_FRAME_POOL = 1;

    using namespace nshogi;
    using namespace nshogi::engine;
    using namespace nshogi::engine::selfplay;

    // Setup allocator.
    allocator::getNodeAllocator().resize((std::size_t)(0.1 * (double)AVAILABLE_MEMORY));
    allocator::getEdgeAllocator().resize((std::size_t)(0.9 * (double)AVAILABLE_MEMORY));

    // Prepare queue.
    auto SearchQueue = std::make_unique<FrameQueue>();
    auto EvaluationQueue = std::make_unique<FrameQueue>();
    auto SaveQueue = std::make_unique<FrameQueue>();

    // Prepare garbage collectors.
    auto GC = std::make_unique<mcts::GarbageCollector>(1);

    // Prepare empty frames.
    for (std::size_t I = 0; I < NUM_FRAME_POOL; ++I) {
        auto F = std::make_unique<Frame>(GC.get());
        SearchQueue->add(std::move(F));
    }

    // Prepare workers.
    std::vector<std::unique_ptr<worker::Worker>> SearchWorkers;
    for (std::size_t I = 0; I < NUM_SEARCH_WORKERS; ++I) {
        SearchWorkers.emplace_back(
                std::make_unique<Worker>(SearchQueue.get(), EvaluationQueue.get(), SaveQueue.get()));
    }

    // Launch workers.
    for (auto& Worker : SearchWorkers) {
        Worker->start();
    }

    // Wait workers.
    for (auto& Worker : SearchWorkers) {
        Worker->await();
    }

    return 0;
}
