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

// A fixed-size, 4-way set-associative cache that maps a position hash
// to its evaluation (policy over the legal moves, win rate and draw rate).
//
// Layout: the table is one contiguous region that holds an array of
// bucket headers followed by an array of payload slots. A header is
// exactly one cache line and contains everything needed to decide
// hit/miss (keys, move counts, recency stamps and a version counter),
// so probing a bucket costs a single cache-line read; the payload
// (the policy array) is touched only on a hit.
//
// Concurrency: each bucket is a seqlock. load() takes no lock: it reads
// the version, copies the entry, and re-reads the version to detect a
// concurrent writer, returning false (a miss) on interference. store()
// acquires the bucket by making the version odd with a single CAS and
// gives up immediately if another writer owns it. Both operations are
// wait-free and may spuriously fail under contention, which callers
// already treat as a cache miss; a miss never blocks.
class EvalCache {
 public:
    static constexpr std::size_t MAX_CACHE_MOVES_COUNT = 164;

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
    static constexpr std::size_t NUM_WAYS = 4;

    // One cache line per bucket. NumMoves[W] == 0 means way W is empty.
    // Version is the seqlock counter: odd while a writer is active.
    // Recency stamps are replacement hints: a way is stamped 255 when
    // it is inserted or hit, and the other ways decay on each
    // insertion, so the way with the smallest stamp is replaced first.
    // Readers update the stamps without holding the lock, so a stamp
    // may occasionally be applied to a way that a concurrent writer
    // just replaced; that only perturbs a future replacement choice.
    struct alignas(64) Bucket {
        std::atomic<uint64_t> Key[NUM_WAYS];
        std::atomic<uint16_t> NumMoves[NUM_WAYS];
        std::atomic<uint8_t> Recency[NUM_WAYS];
        std::atomic<uint32_t> Version;
    };

    struct Payload {
        float WinRate;
        float DrawRate;
        float Policy[MAX_CACHE_MOVES_COUNT];
    };

    static_assert(sizeof(Bucket) == 64,
                  "a bucket header must be exactly one cache line");
    static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                      std::atomic<uint32_t>::is_always_lock_free &&
                      std::atomic<uint16_t>::is_always_lock_free &&
                      std::atomic<uint8_t>::is_always_lock_free,
                  "the seqlock scheme requires lock-free atomics");

    static constexpr std::size_t BUCKET_BYTES =
        sizeof(Bucket) + NUM_WAYS * sizeof(Payload);

    std::size_t bucketIndex(uint64_t Hash) const;

    Payload* payloadOf(std::size_t BucketIdx, std::size_t Way) const {
        return Payloads + BucketIdx * NUM_WAYS + Way;
    }

    const std::size_t NumBuckets;
    const std::size_t RegionBytes;
    void* const Region;
    Bucket* const Buckets;
    Payload* const Payloads;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
