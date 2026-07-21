//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "evalcache.h"

#include <algorithm>
#include <cstring>
#include <new>

#if defined(__linux__)

#include <sys/mman.h>

#endif

namespace nshogi {
namespace engine {
namespace mcts {

namespace {

std::size_t computeNumBuckets(std::size_t MemorySizeMB,
                              std::size_t BucketBytes) {
    const std::size_t N = std::max<std::size_t>(1, MemorySizeMB * 1024ULL *
                                                       1024ULL / BucketBytes);
    // bucketIndex() requires the bucket count to fit in 32 bits.
    return std::min<std::size_t>(N, 0xFFFFFFFFULL);
}

void* allocateRegion(std::size_t Bytes) {
#if defined(__linux__)
    // Anonymous pages are zero-filled and are committed only when they
    // are first touched, so constructing a large cache is cheap and
    // memory is consumed on demand. Huge pages reduce TLB misses of
    // the random probes into the table.
    void* Ptr = ::mmap(nullptr, Bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (Ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    ::madvise(Ptr, Bytes, MADV_HUGEPAGE);
    return Ptr;
#else
    return ::operator new(Bytes, std::align_val_t{64});
#endif
}

void freeRegion(void* Ptr, [[maybe_unused]] std::size_t Bytes) {
#if defined(__linux__)
    ::munmap(Ptr, Bytes);
#else
    ::operator delete(Ptr, std::align_val_t{64});
#endif
}

} // namespace

EvalCache::EvalCache(std::size_t MemorySize)
    : NumBuckets(computeNumBuckets(MemorySize, BUCKET_BYTES))
    , RegionBytes(NumBuckets * BUCKET_BYTES)
    , Region(allocateRegion(RegionBytes))
    , Buckets(static_cast<Bucket*>(Region))
    , Payloads(reinterpret_cast<Payload*>(static_cast<char*>(Region) +
                                          NumBuckets * sizeof(Bucket))) {
    // Zero-initialized headers (all ways empty, version 0) are the
    // valid initial state. The payload region needs no initialization:
    // a payload slot is read only after a completed store() published
    // the corresponding way. std::atomic is not an implicit-lifetime
    // type, so the headers are constructed explicitly.
    for (std::size_t I = 0; I < NumBuckets; ++I) {
        new (&Buckets[I]) Bucket;
    }
}

EvalCache::~EvalCache() {
    freeRegion(Region, RegionBytes);
}

std::size_t EvalCache::bucketIndex(uint64_t Hash) const {
    // Mix the hash first: the bucket spread must not depend on how
    // core::State::getHash() distributes its bits across the word
    // (the side-to-move bit and the stands occupy fixed positions,
    // and the Zobrist keys' width is not guaranteed here). Only the
    // bucket selection uses the mixed value; the ways store the raw
    // hash as the key. The mixer is the 64-bit finalizer of
    // MurmurHash3.
    uint64_t X = Hash;
    X ^= X >> 33;
    X *= 0xFF51AFD7ED558CCDULL;
    X ^= X >> 33;
    X *= 0xC4CEB9FE1A85EC53ULL;
    X ^= X >> 33;
    // Multiply-shift maps the mixed value onto [0, NumBuckets) without
    // an integer division. NumBuckets fits in 32 bits.
    return (std::size_t)(((X >> 32) * (uint64_t)NumBuckets) >> 32);
}

bool EvalCache::store(uint64_t Hash, uint16_t NumM, const float* P, float WR,
                      float D) {
    if (NumM == 0 || NumM > MAX_CACHE_MOVES_COUNT) {
        // An entry with no moves carries no reusable information
        // (loaders compare NumMoves against a nonzero legal-move
        // count), and NumMoves == 0 marks an empty way internally.
        return false;
    }

    const std::size_t BIdx = bucketIndex(Hash);
    Bucket* B = &Buckets[BIdx];

    uint32_t V = B->Version.load(std::memory_order_relaxed);
    if ((V & 1U) != 0) {
        return false;
    }
    if (!B->Version.compare_exchange_strong(V, V + 1, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
        // Another writer owns this bucket.
        return false;
    }

    // The version is now odd: this thread owns the bucket.
    // Pick the way to write to: an empty way if there is one,
    // otherwise the least recently used one.
    std::size_t ReplaceWay = 0;
    std::size_t Empty = NUM_WAYS;
    uint8_t MinRecency = 255;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        const uint16_t NM = B->NumMoves[W].load(std::memory_order_relaxed);

        if (B->Key[W].load(std::memory_order_relaxed) == Hash && NM == NumM) {
            // The entry already exists; keep the stored values and only
            // refresh its recency.
            B->Recency[W].store(255, std::memory_order_relaxed);
            B->Version.store(V + 2, std::memory_order_release);
            return true;
        }

        if (NM == 0) {
            if (Empty == NUM_WAYS) {
                Empty = W;
            }
        } else {
            const uint8_t R = B->Recency[W].load(std::memory_order_relaxed);
            if (R < MinRecency) {
                MinRecency = R;
                ReplaceWay = W;
            }
        }
    }
    if (Empty != NUM_WAYS) {
        ReplaceWay = Empty;
    }

    Payload* PL = payloadOf(BIdx, ReplaceWay);
    PL->WinRate = WR;
    PL->DrawRate = D;
    std::memcpy(PL->Policy, P, sizeof(float) * NumM);

    B->Key[ReplaceWay].store(Hash, std::memory_order_relaxed);
    B->NumMoves[ReplaceWay].store(NumM, std::memory_order_relaxed);
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (W == ReplaceWay) {
            B->Recency[W].store(255, std::memory_order_relaxed);
        } else {
            const uint8_t R = B->Recency[W].load(std::memory_order_relaxed);
            B->Recency[W].store((uint8_t)(R >> 1), std::memory_order_relaxed);
        }
    }

    B->Version.store(V + 2, std::memory_order_release);
    return true;
}

bool EvalCache::load(uint64_t Hash, EvalInfo* EI) {
    const std::size_t BIdx = bucketIndex(Hash);
    Bucket* B = &Buckets[BIdx];

    const uint32_t V0 = B->Version.load(std::memory_order_acquire);
    if ((V0 & 1U) != 0) {
        // A writer is updating this bucket.
        return false;
    }

    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (B->Key[W].load(std::memory_order_relaxed) != Hash) {
            continue;
        }

        const uint16_t NumM = B->NumMoves[W].load(std::memory_order_relaxed);
        if (NumM == 0 || NumM > MAX_CACHE_MOVES_COUNT) {
            // An empty way whose zero key happened to match.
            continue;
        }

        // The payload is copied without holding a lock, so it may be
        // torn by a concurrent store(); the version re-check below
        // discards such a read. (This is the usual seqlock pattern;
        // the racing reads are benign and are never returned.)
        const Payload* PL = payloadOf(BIdx, W);
        EI->NumMoves = NumM;
        EI->WinRate = PL->WinRate;
        EI->DrawRate = PL->DrawRate;
        std::memcpy(EI->Policy, PL->Policy, sizeof(float) * NumM);

        std::atomic_thread_fence(std::memory_order_acquire);
        if (B->Version.load(std::memory_order_relaxed) != V0) {
            // A writer interleaved with the copy.
            return false;
        }

        // Mark the way as recently used (a replacement hint only).
        B->Recency[W].store(255, std::memory_order_relaxed);
        return true;
    }

    return false;
}

bool EvalCache::load(const core::State& St, EvalInfo* EI) {
    return load(St.getHash(), EI);
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
