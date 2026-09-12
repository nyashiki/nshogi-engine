//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
#define NSHOGI_ENGINE_MCTS_EVALCACHE_H

#include <atomic>
#include <cinttypes>
#include <cstddef>

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace mcts {

class EvalCache {
 public:
    static constexpr std::size_t MAX_CACHE_MOVES_COUNT = 600;

    struct EvalInfo {
        uint16_t NumMoves;
        float Policy[MAX_CACHE_MOVES_COUNT];
        float WinRate;
        float DrawRate;
    };

    EvalCache(std::size_t MemorySize);
    ~EvalCache();

    EvalCache(const EvalCache&) = delete;
    EvalCache& operator=(const EvalCache&) = delete;

    bool store(uint64_t Hash, uint16_t NumM, const float* P, float WR, float D);
    bool load(uint64_t Hash, EvalInfo*);
    bool load(const core::State&, EvalInfo*);

 private:
    static constexpr std::size_t NUM_WAYS = 8;

    // Two cache lines per bucket. NumMoves[W] == 0 means way W is
    // empty. Offset[W] is the byte offset of way W's entry inside the
    // bucket's payload region. A writer excludes all readers; readers
    // can copy entries concurrently. Lock attempts never wait or retry.
    // Recency stamps are replacement hints: a way is stamped 255 when
    // it is inserted or hit, and the other ways decay on each
    // insertion, so the way with the smallest stamp is replaced first.
    // Keys remain atomic for the scan before locking. Recency stamps
    // remain atomic because multiple readers can refresh the same way.
    struct alignas(128) Bucket {
        std::atomic<uint64_t> Key[NUM_WAYS];
        uint16_t NumMoves[NUM_WAYS];
        uint16_t Offset[NUM_WAYS];
        std::atomic<uint8_t> Recency[NUM_WAYS];
        // Bit 0 denotes the writer; the remaining bits count readers.
        std::atomic<uint32_t> Access;

        bool isWriting() const {
            return (Access.load(std::memory_order_relaxed) & 1U) != 0;
        }

        bool tryLock() {
            uint32_t Expected = 0;
            return Access.compare_exchange_strong(
                Expected, 1, std::memory_order_acquire,
                std::memory_order_relaxed);
        }

        bool tryLockShared() {
            const uint32_t Previous =
                Access.fetch_add(2, std::memory_order_acquire);
            if ((Previous & 1U) != 0) {
                Access.fetch_sub(2, std::memory_order_release);
                return false;
            }
            return true;
        }

        void unlock() {
            // A rejected reader may still be undoing its increment.
            // Clear only the writer bit, preserving those reader counts.
            Access.fetch_sub(1, std::memory_order_release);
        }

        void unlockShared() {
            Access.fetch_sub(2, std::memory_order_release);
        }
    };

    // An entry inside a payload region: WinRate, DrawRate, then
    // NumMoves floats of policy. Everything is 4-byte aligned, so
    // offsets and sizes are multiples of 4.
    static constexpr std::size_t entryBytes(std::size_t NumM) {
        return 2 * sizeof(float) + sizeof(float) * NumM;
    }

    // The payload region of one bucket. Eight independently drawn
    // entries (mean legal-move count 49, mean entry 205 bytes) sum to
    // at most 2432 bytes ~99% of the time, and the largest legal
    // entry (600 moves, 2408 bytes) fits alone.
    static constexpr std::size_t REGION_BYTES = 2432;

    // (entryBytes() is not usable in a constant expression until the
    // class is complete, so the size of the largest entry is spelled
    // out.)
    static_assert(REGION_BYTES >=
                      2 * sizeof(float) + sizeof(float) * MAX_CACHE_MOVES_COUNT,
                  "a region must be able to hold the largest entry");
    static_assert(REGION_BYTES % 4 == 0 && REGION_BYTES <= UINT16_MAX,
                  "offsets are 4-byte-aligned uint16 byte counts");
    static_assert(sizeof(Bucket) == 128,
                  "a bucket header must be exactly two cache lines");
    static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                      std::atomic<uint32_t>::is_always_lock_free &&
                      std::atomic<uint8_t>::is_always_lock_free,
                  "bucket access requires lock-free atomics");

    static constexpr std::size_t BUCKET_BYTES = sizeof(Bucket) + REGION_BYTES;

    std::size_t bucketIndex(uint64_t Hash) const;

    char* regionOf(std::size_t BucketIdx) const {
        return Regions + BucketIdx * REGION_BYTES;
    }

    const std::size_t NumBuckets;
    const std::size_t RegionBytes;
    void* const Region;
    Bucket* const Buckets;
    char* const Regions;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
