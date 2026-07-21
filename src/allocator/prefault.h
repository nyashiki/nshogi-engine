//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_ALLOCATOR_PREFAULT_H
#define NSHOGI_ENGINE_ALLOCATOR_PREFAULT_H

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#if defined(__linux__)

#include <sys/mman.h>

#endif

namespace nshogi {
namespace engine {
namespace allocator {

// Run Fn(Begin, End) over [0, Count) split into contiguous slices, one
// slice per thread. Used to parallelize resize()-time initialization,
// which is memory-bound (page faults and stores), so the slice count
// is capped by the hardware concurrency; slices smaller than
// MinPerThread are not worth a thread.
template <typename F>
inline void forEachSliceParallel(std::size_t Count, std::size_t MinPerThread,
                                 F&& Fn) {
    const std::size_t HW =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t NumThreads =
        std::min(HW, std::max<std::size_t>(
                         1, MinPerThread == 0 ? HW : Count / MinPerThread));

    if (NumThreads <= 1) {
        Fn(std::size_t{0}, Count);
        return;
    }

    const std::size_t PerThread = (Count + NumThreads - 1) / NumThreads;
    std::vector<std::thread> Threads;
    Threads.reserve(NumThreads);
    for (std::size_t T = 0; T < NumThreads; ++T) {
        const std::size_t Begin = std::min(Count, T * PerThread);
        const std::size_t End = std::min(Count, Begin + PerThread);
        if (Begin >= End) {
            break;
        }
        Threads.emplace_back([&Fn, Begin, End]() { Fn(Begin, End); });
    }
    for (auto& Th : Threads) {
        Th.join();
    }
}

// Commit all pages of [Memory, Memory + Bytes) so that no page fault
// happens once the region is in use. Equivalent to MAP_POPULATE, but
// runs on all cores: the kernel populates (zero-fills) the pages of a
// mapping sequentially, which makes MAP_POPULATE the dominant cost of
// resize() for multi-GB regions.
inline void prefaultRegionParallel(void* Memory, std::size_t Bytes) {
    constexpr std::size_t SliceBytes = 32ULL << 20;
    forEachSliceParallel(Bytes, SliceBytes,
                         [&](std::size_t Begin, std::size_t End) {
                             char* P = static_cast<char*>(Memory) + Begin;
                             const std::size_t Len = End - Begin;
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
                             if (::madvise(P, Len, MADV_POPULATE_WRITE) == 0) {
                                 return;
                             }
#endif
                             // Fallback (madvise unsupported): touch one byte
                             // per page. The pages are demand-zeroed, so
                             // writing zero does not change the contents.
                             for (std::size_t I = 0; I < Len; I += 4096) {
                                 static_cast<volatile char*>(P)[I] = 0;
                             }
                         });
}

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_PREFAULT_H
