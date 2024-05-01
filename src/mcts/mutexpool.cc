#include "mutexpool.h"
#include <cstddef>
#include <mutex>


namespace nshogi {
namespace engine {
namespace mcts {

MutexPool::MutexPool(std::size_t PoolSize): Size(PoolSize) {
    Pool = std::make_unique<std::mutex[]>(PoolSize);
}

std::mutex* MutexPool::get(void* Ptr) {
    return Pool.get() + (std::size_t)Ptr % Size;
}

std::mutex* MutexPool::getRootMtx() {
    return &RootMtx;
}


} // namespace mcts
} // namespace engine
} // namespace nshogi
