//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "mutexpool.h"
#include "../lock/spinlock.h"

#include <cstddef>
#include <mutex>


namespace nshogi {
namespace engine {
namespace mcts {

template <typename LockType>
MutexPool<LockType>::MutexPool(std::size_t PoolSize): Size(PoolSize) {
    Pool = std::make_unique<LockType[]>(PoolSize);
}

template <typename LockType>
LockType* MutexPool<LockType>::get(void* Ptr) {
    return Pool.get() + (std::size_t)Ptr % Size;
}

template <typename LockType>
LockType* MutexPool<LockType>::getRootMtx() {
    return &RootMtx;
}

template class MutexPool<std::mutex>;
template class MutexPool<lock::SpinLock>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
