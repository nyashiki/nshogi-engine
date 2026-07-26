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
    // bucket's payload region. Version is the seqlock counter: odd
    // while a writer is active.
    // Recency stamps are replacement hints: a way is stamped 255 when
    // it is inserted or hit, and the other ways decay on each
    // insertion, so the way with the smallest stamp is replaced first.
    // Readers update the stamps without holding the lock, so a stamp
    // may occasionally be applied to a way that a concurrent writer
    // just replaced; that only perturbs a future replacement choice.
    struct alignas(128) Bucket {
        std::atomic<uint64_t> Key[NUM_WAYS];
        std::atomic<uint16_t> NumMoves[NUM_WAYS];
        std::atomic<uint16_t> Offset[NUM_WAYS];
        std::atomic<uint8_t> Recency[NUM_WAYS];
        std::atomic<uint32_t> Version;
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
                      std::atomic<uint16_t>::is_always_lock_free &&
                      std::atomic<uint8_t>::is_always_lock_free,
                  "the seqlock scheme requires lock-free atomics");

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
