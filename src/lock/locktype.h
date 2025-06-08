//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_LOCK_LOCKTYPE_H
#define NSHOGI_ENGINE_LOCK_LOCKTYPE_H

namespace nshogi {
namespace engine {
namespace lock {

template <typename T>
concept LockType = requires(T& Lock) {
    Lock.lock();
    Lock.unlock();
};

} // namespace lock
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_LOCK_LOCKTYPE_H
