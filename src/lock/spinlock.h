//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_LOCK_SPIN_LOCK_H
#define NSHOGI_ENGINE_LOCK_SPIN_LOCK_H

#include <atomic>
#include <thread>

#ifdef USE_SSE2

#include <emmintrin.h>

#endif

namespace nshogi {
namespace engine {
namespace lock {

class SpinLock {
 public:
    SpinLock() {
        Flag.clear(std::memory_order_release);
    }

    void lock() {
        uint_fast8_t StreakFail = 0;
        while (Flag.test_and_set(std::memory_order_acquire)) {
            if (StreakFail < 4) {
                ++StreakFail;
            } else {
#ifdef USE_SSE2
            _mm_pause();
#else
            std::this_thread::yield();
#endif
            }
        }
    }

    void unlock() {
        Flag.clear(std::memory_order_release);
    }

 private:
    std::atomic_flag Flag;
};

} // namespace lock
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_LOCK_SPIN_LOCK_H
