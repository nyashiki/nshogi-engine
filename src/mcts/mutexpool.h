//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
#define NSHOGI_ENGINE_MCTS_MUTEXPOOL_H

#include "../lock/locktype.h"
#include "../lock/spinlock.h"

#include <cstddef>
#include <memory>
#include <mutex>

namespace nshogi {
namespace engine {
namespace mcts {

template <lock::LockType LockT = lock::SpinLock>
class MutexPool {
 public:
    MutexPool(std::size_t PoolMemorySize);

    LockT* get(uint64_t Hash);
    LockT* getRootMtx();

    using LockType = LockT;

 private:
    const std::size_t Size;
    std::unique_ptr<LockT[]> Pool;
    LockT RootMtx;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_MUTEXPOOL_H
