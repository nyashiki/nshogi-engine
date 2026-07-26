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
    , Regions(static_cast<char*>(Region) + NumBuckets * sizeof(Bucket)) {
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
    // Multiply-shift maps the hash's top 32 bits onto [0, NumBuckets).
    return (std::size_t)(((Hash >> 32) * (uint64_t)NumBuckets) >> 32);
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

    // The version is now odd: this thread owns the bucket, so the
    // header fields and the payload region are safe to mutate; readers
    // that overlap with the mutation discard their copy on the version
    // re-check.
    uint16_t NM[NUM_WAYS];
    uint16_t Off[NUM_WAYS];
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        NM[W] = B->NumMoves[W].load(std::memory_order_relaxed);
        Off[W] = B->Offset[W].load(std::memory_order_relaxed);

        if (NM[W] != 0 && NM[W] == NumM &&
            B->Key[W].load(std::memory_order_relaxed) == Hash) {
            // The entry already exists; keep the stored values and only
            // refresh its recency.
            B->Recency[W].store(255, std::memory_order_relaxed);
            B->Version.store(V + 2, std::memory_order_release);
            return true;
        }
    }

    const std::size_t Need = entryBytes(NumM);

    // Make room: the new entry needs a free way and Need bytes of
    // region space. Evict the least recently used way first when all
    // four are occupied, then keep evicting until the entry fits.
    // The loop terminates because Need <= REGION_BYTES.
    std::size_t LiveBytes = 0;
    bool HasFreeWay = false;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (NM[W] != 0) {
            LiveBytes += entryBytes(NM[W]);
        } else {
            HasFreeWay = true;
        }
    }

    const auto EvictLRU = [&]() {
        std::size_t Victim = NUM_WAYS;
        uint8_t MinRecency = 0;
        for (std::size_t W = 0; W < NUM_WAYS; ++W) {
            if (NM[W] == 0) {
                continue;
            }
            const uint8_t R = B->Recency[W].load(std::memory_order_relaxed);
            if (Victim == NUM_WAYS || R < MinRecency) {
                MinRecency = R;
                Victim = W;
            }
        }
        LiveBytes -= entryBytes(NM[Victim]);
        NM[Victim] = 0;
        B->NumMoves[Victim].store(0, std::memory_order_relaxed);
    };

    if (!HasFreeWay) {
        EvictLRU();
    }
    while (REGION_BYTES - LiveBytes < Need) {
        EvictLRU();
    }

    // Place the entry after the last live one if the tail is large
    // enough, or failing that into the first eviction gap it fits in;
    // only when neither works are the live entries compacted to the
    // front of the region. Compaction moves at most REGION_BYTES bytes
    // inside one bucket and happens at most once per insertion.
    char* Rg = regionOf(BIdx);

    std::size_t Order[NUM_WAYS];
    std::size_t NumLive = 0;
    std::size_t End = 0;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (NM[W] != 0) {
            Order[NumLive++] = W;
            End = std::max<std::size_t>(End, Off[W] + entryBytes(NM[W]));
        }
    }

    std::size_t At = End;
    if (REGION_BYTES - End < Need) {
        std::sort(Order, Order + NumLive, [&](std::size_t A, std::size_t C) {
            return Off[A] < Off[C];
        });

        // Gaps precede each live entry in offset order.
        At = REGION_BYTES; // Sentinel: no gap found yet.
        std::size_t Cursor = 0;
        for (std::size_t I = 0; I < NumLive; ++I) {
            const std::size_t W = Order[I];
            if (Off[W] - Cursor >= Need) {
                At = Cursor;
                break;
            }
            Cursor = Off[W] + entryBytes(NM[W]);
        }

        if (At == REGION_BYTES) {
            Cursor = 0;
            for (std::size_t I = 0; I < NumLive; ++I) {
                const std::size_t W = Order[I];
                const std::size_t Sz = entryBytes(NM[W]);
                if (Off[W] != Cursor) {
                    std::memmove(Rg + Cursor, Rg + Off[W], Sz);
                    B->Offset[W].store((uint16_t)Cursor,
                                       std::memory_order_relaxed);
                }
                Cursor += Sz;
            }
            At = Cursor;
        }
    }

    std::memcpy(Rg + At, &WR, sizeof(float));
    std::memcpy(Rg + At + sizeof(float), &D, sizeof(float));
    std::memcpy(Rg + At + 2 * sizeof(float), P, sizeof(float) * NumM);

    std::size_t Slot = NUM_WAYS;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (NM[W] == 0) {
            Slot = W;
            break;
        }
    }
    B->Key[Slot].store(Hash, std::memory_order_relaxed);
    B->NumMoves[Slot].store(NumM, std::memory_order_relaxed);
    B->Offset[Slot].store((uint16_t)At, std::memory_order_relaxed);
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (W == Slot) {
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

        const uint16_t Off = B->Offset[W].load(std::memory_order_relaxed);
        if (Off > REGION_BYTES - entryBytes(NumM)) {
            // A torn read of the header (the offset and the move count
            // belong to different versions of the way). Bounding the
            // copy keeps it inside this bucket's region; the version
            // re-check below rejects the result anyway.
            continue;
        }

        // The payload is copied without holding a lock, so it may be
        // torn by a concurrent store(); the version re-check below
        // discards such a read. (This is the usual seqlock pattern;
        // the racing reads are benign and are never returned.)
        const char* Rg = regionOf(BIdx);
        EI->NumMoves = NumM;
        std::memcpy(&EI->WinRate, Rg + Off, sizeof(float));
        std::memcpy(&EI->DrawRate, Rg + Off + sizeof(float), sizeof(float));
        std::memcpy(EI->Policy, Rg + Off + 2 * sizeof(float),
                    sizeof(float) * NumM);

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
