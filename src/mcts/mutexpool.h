#ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
#define NSHOGI_ENGINE_MCTS_MUTEXPOOL_H


#include <cstddef>
#include <mutex>
#include <memory>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename LockType = std::mutex>
class MutexPool {
 public:
    MutexPool(std::size_t PoolSize);

    LockType* get(void* Ptr);
    LockType* getRootMtx();

 private:
    const std::size_t Size;
    std::unique_ptr<LockType[]> Pool;
    LockType RootMtx;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
