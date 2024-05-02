#ifndef NSHOGI_ENGINE_BENCH_MCTS_H
#define NSHOGI_ENGINE_BENCH_MCTS_H

#include <cinttypes>

namespace nshogi {
namespace engine {
namespace bench {

void benchMCTS(uint64_t DurationSeconds, std::size_t BatchSize, std::size_t NumGPUs, std::size_t NumThreadsPerGPU, std::size_t NumCheckmateSearchers, std::size_t EvalCacheMB);

} // namespace bench
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BENCH_MCTS_H
