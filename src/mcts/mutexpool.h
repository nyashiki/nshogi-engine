#ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
#define NSHOGI_ENGINE_MCTS_MUTEXPOOL_H


#include <cstddef>
#include <mutex>
#include <memory>

namespace nshogi {
namespace engine {
namespace mcts {

class MutexPool {
 public:
    MutexPool(std::size_t PoolSize);

    std::mutex* get(void* Ptr);
    std::mutex* getRootMtx();

 private:
    const std::size_t Size;
    std::unique_ptr<std::mutex[]> Pool;
    std::mutex RootMtx;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
