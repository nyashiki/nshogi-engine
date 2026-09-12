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
#include <memory>
#include <vector>

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace mcts {

class EvalCache {
 public:
    struct EvalInfo {
        uint16_t NumMoves = 0;
        // Reusable scratch storage; only the first NumMoves values are valid.
        // Retaining its size avoids reinitializing overwritten values on load.
        std::vector<float> Policy;
        float WinRate = 0.0f;
        float DrawRate = 0.0f;
    };

    EvalCache(std::size_t MemorySize);
    ~EvalCache();

    EvalCache(const EvalCache&) = delete;
    EvalCache& operator=(const EvalCache&) = delete;

    bool store(uint64_t Hash, uint16_t NumM, const float* P, float WR, float D);
    bool load(uint64_t Hash, EvalInfo*);
    bool load(const core::State&, EvalInfo*);

    // Retained storage, including headers and a conservative allocation
    // overhead allowance. Transient resize buffers / caller scratch are extra.
    std::size_t getMemoryUsed() const;
    std::size_t getMemoryLimit() const;

 private:
    static constexpr std::size_t NUM_WAYS = 8;

    // Entries are packed in way order in a dynamically sized buffer. Only
    // NumMoves[W] + 2 floats are stored for an occupied way; zero means empty.
    // Readers share the bucket lock, while a writer can resize or free its
    // buffer only after all readers have left. Keys alone are probed unlocked.
    struct alignas(128) Bucket {
        std::atomic<uint64_t> Key[NUM_WAYS];
        uint16_t NumMoves[NUM_WAYS];
        // Float offsets; UINT16_MAX asks offsetOf() to compute a wide offset.
        uint16_t Offset[NUM_WAYS];
        std::atomic<uint8_t> Recency[NUM_WAYS];
        // Bit 0 denotes the writer; the remaining bits count readers.
        std::atomic<uint32_t> Access;
        std::unique_ptr<float[]> Data;
        uint32_t Size;
        uint32_t Capacity;

        bool isWriting() const {
            return (Access.load(std::memory_order_relaxed) & 1U) != 0;
        }

        bool try_lock() {
            uint32_t Expected = 0;
            return Access.compare_exchange_strong(Expected, 1,
                                                  std::memory_order_acquire,
                                                  std::memory_order_relaxed);
        }

        bool try_lock_shared() {
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

        void unlock_shared() {
            Access.fetch_sub(2, std::memory_order_release);
        }
    };

    static constexpr std::size_t entryFloats(uint16_t NumM) {
        return NumM == 0 ? 0 : std::size_t{NumM} + 2;
    }

    static std::size_t allocationBytes(std::size_t NumFloats);
    static std::size_t computeMemoryLimit(std::size_t MemoryMB);
    static std::size_t computeNumBuckets(std::size_t MemoryBytes);
    static std::size_t offsetOf(const Bucket*, std::size_t Way);
    static void updateOffsets(Bucket*);
    static std::size_t leastRecent(const Bucket*, std::size_t ExcludedWay);
    static void erase(Bucket*, std::size_t Way);

    std::size_t bucketIndex(uint64_t Hash) const;
    bool reserve(std::size_t Bytes);
    bool reclaimAndReserve(std::size_t Bytes, std::size_t ExcludedBucket);

    static_assert(sizeof(Bucket) == 128,
                  "a bucket header must be exactly two cache lines");
    static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                      std::atomic<uint32_t>::is_always_lock_free &&
                      std::atomic<uint8_t>::is_always_lock_free &&
                      std::atomic<std::size_t>::is_always_lock_free,
                  "bucket access requires lock-free atomics");

    const std::size_t MemoryLimit;
    const std::size_t NumBuckets;
    const std::size_t HeaderBytes;
    const std::size_t PayloadLimit;
    Bucket* const Buckets;
    // Keep accounting writes off the cache lines read by every lookup.
    alignas(64) std::atomic<std::size_t> PayloadBytes{0};
    alignas(64) std::atomic<std::size_t> EvictionCursor{0};
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
