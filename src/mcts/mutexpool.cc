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

namespace nshogi {
namespace engine {
namespace mcts {

template <lock::LockType LockT>
MutexPool<LockT>::MutexPool(std::size_t PoolMemorySize)
    : Size(PoolMemorySize / sizeof(LockT)) {
    Pool = std::make_unique<LockT[]>(Size);
}

template <lock::LockType LockT>
LockT* MutexPool<LockT>::get(uint64_t Hash) {
    return Pool.get() + (std::size_t)Hash % Size;
}

template <lock::LockType LockT>
LockT* MutexPool<LockT>::getRootMtx() {
    return &RootMtx;
}

template class MutexPool<std::mutex>;
template class MutexPool<lock::SpinLock>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
