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
#include <cassert>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <shared_mutex>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace nshogi {
namespace engine {
namespace mcts {

std::size_t EvalCache::allocationBytes(std::size_t NumFloats) {
    if (NumFloats == 0) {
        return 0;
    }
    constexpr std::size_t Alignment = alignof(std::max_align_t);
    return ((NumFloats * sizeof(float) + Alignment - 1) & ~(Alignment - 1)) +
           2 * sizeof(std::size_t);
}

std::size_t EvalCache::computeMemoryLimit(std::size_t MemoryMB) {
    constexpr std::size_t MiB = 1024 * 1024;
    if (MemoryMB > std::numeric_limits<std::size_t>::max() / MiB) {
        throw std::bad_alloc();
    }
    return std::max<std::size_t>(4096, MemoryMB * MiB);
}

std::size_t EvalCache::computeNumBuckets(std::size_t MemoryBytes) {
    // Preserve the previous number of hash ways for a given memory setting.
    // This is only an index-sizing estimate, not a per-entry allocation or
    // size limit. Actual payload storage is lazy and budgeted independently.
    constexpr std::size_t EstimatedBytesPerWay = 320;
    constexpr std::size_t EstimatedBucketBytes =
        NUM_WAYS * EstimatedBytesPerWay;
    return std::clamp<std::size_t>(MemoryBytes / EstimatedBucketBytes, 1,
                                   UINT32_MAX);
}

EvalCache::EvalCache(std::size_t MemorySize)
    : MemoryLimit(computeMemoryLimit(MemorySize))
    , NumBuckets(computeNumBuckets(MemoryLimit))
    , HeaderBytes(NumBuckets * sizeof(Bucket))
    , PayloadLimit(MemoryLimit - HeaderBytes)
    , Buckets([this]() -> Bucket* {
#if defined(__linux__)
        void* Ptr = ::mmap(nullptr, HeaderBytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (Ptr == MAP_FAILED) {
            throw std::bad_alloc();
        }
        ::madvise(Ptr, HeaderBytes, MADV_HUGEPAGE);
#else
        void* Ptr = ::operator new(HeaderBytes,
                                   std::align_val_t{alignof(Bucket)});
#endif
        return static_cast<Bucket*>(Ptr);
    }()) {
    for (std::size_t I = 0; I < NumBuckets; ++I) {
        new (&Buckets[I]) Bucket{};
    }
}

EvalCache::~EvalCache() {
    for (std::size_t I = 0; I < NumBuckets; ++I) {
        Buckets[I].~Bucket();
    }
#if defined(__linux__)
    ::munmap(Buckets, HeaderBytes);
#else
    ::operator delete(Buckets, std::align_val_t{alignof(Bucket)});
#endif
}

std::size_t EvalCache::getMemoryUsed() const {
    return HeaderBytes + PayloadBytes.load(std::memory_order_relaxed);
}

std::size_t EvalCache::getMemoryLimit() const {
    return MemoryLimit;
}

std::size_t EvalCache::bucketIndex(uint64_t Hash) const {
    return (std::size_t)(((Hash >> 32) * (uint64_t)NumBuckets) >> 32);
}

std::size_t EvalCache::offsetOf(const Bucket* B, std::size_t Way) {
    if (B->Offset[Way] != UINT16_MAX) {
        return B->Offset[Way];
    }
    // Very large policies can exceed a compact offset. Falling back keeps
    // the header small without imposing a limit on the number of moves.
    std::size_t Offset = 0;
    for (std::size_t I = 0; I < Way; ++I) {
        Offset += entryFloats(B->NumMoves[I]);
    }
    return Offset;
}

void EvalCache::updateOffsets(Bucket* B) {
    std::size_t Offset = 0;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        B->Offset[W] = (uint16_t)std::min<std::size_t>(Offset, UINT16_MAX);
        Offset += entryFloats(B->NumMoves[W]);
    }
}

std::size_t EvalCache::leastRecent(const Bucket* B, std::size_t ExcludedWay) {
    std::size_t Victim = NUM_WAYS;
    uint8_t MinRecency = 0;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (W == ExcludedWay || B->NumMoves[W] == 0) {
            continue;
        }
        const uint8_t R = B->Recency[W].load(std::memory_order_relaxed);
        if (Victim == NUM_WAYS || R < MinRecency) {
            MinRecency = R;
            Victim = W;
        }
    }
    return Victim;
}

void EvalCache::erase(Bucket* B, std::size_t Way) {
    const std::size_t Offset = offsetOf(B, Way);
    const std::size_t Size = entryFloats(B->NumMoves[Way]);
    std::memmove(B->Data.get() + Offset, B->Data.get() + Offset + Size,
                 (B->Size - Offset - Size) * sizeof(float));
    B->Size -= (uint32_t)Size;
    B->NumMoves[Way] = 0;
    updateOffsets(B);
}

bool EvalCache::reserve(std::size_t Bytes) {
    std::size_t Previous = PayloadBytes.load(std::memory_order_relaxed);
    while (Bytes <= PayloadLimit - Previous) {
        if (PayloadBytes.compare_exchange_weak(
                Previous, Previous + Bytes, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool EvalCache::reclaimAndReserve(std::size_t Bytes,
                                  std::size_t ExcludedBucket) {
    if (reserve(Bytes)) {
        return true;
    }
    // Used only when a new entry cannot fit even after local evictions.
    // An oversized entry can borrow space from other buckets; there is no
    // fixed per-bucket payload limit. Never wait on another bucket's readers.
    for (std::size_t I = 0; I < NumBuckets; ++I) {
        const std::size_t Idx =
            EvictionCursor.fetch_add(1, std::memory_order_relaxed) % NumBuckets;
        if (Idx == ExcludedBucket) {
            continue;
        }
        Bucket* B = &Buckets[Idx];
        std::unique_lock Guard(*B, std::try_to_lock);
        if (!Guard.owns_lock() || B->Capacity == 0) {
            continue;
        }
        const std::size_t ReleasedBytes = allocationBytes(B->Capacity);
        B->Data.reset();
        B->Capacity = B->Size = 0;
        for (std::size_t W = 0; W < NUM_WAYS; ++W) {
            B->NumMoves[W] = 0;
            B->Offset[W] = 0;
        }
        PayloadBytes.fetch_sub(ReleasedBytes, std::memory_order_relaxed);
        if (reserve(Bytes)) {
            return true;
        }
    }
    return false;
}

bool EvalCache::store(uint64_t Hash, uint16_t NumM, const float* P, float WR,
                      float D) {
    if (NumM == 0) {
        return false;
    }
    const std::size_t NewSize = entryFloats(NumM);
    if (allocationBytes(NewSize) > PayloadLimit) {
        return false;
    }
    const std::size_t BIdx = bucketIndex(Hash);
    Bucket* B = &Buckets[BIdx];
    std::unique_lock Guard(*B, std::try_to_lock);
    if (!Guard.owns_lock()) {
        return false;
    }

    std::size_t Slot = NUM_WAYS;
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (B->NumMoves[W] == NumM &&
            B->Key[W].load(std::memory_order_relaxed) == Hash) {
            B->Recency[W].store(255, std::memory_order_relaxed);
            return true;
        }
        if (B->NumMoves[W] == 0 && Slot == NUM_WAYS) {
            Slot = W;
        }
    }
    if (Slot == NUM_WAYS) {
        Slot = leastRecent(B, NUM_WAYS);
    }

    const std::size_t OldBytes = allocationBytes(B->Capacity);
    std::size_t Need;
    std::size_t NewCapacity;
    std::size_t NewBytes;
    while (true) {
        Need = B->Size - entryFloats(B->NumMoves[Slot]) + NewSize;
        NewCapacity = B->Capacity;
        if (Need > B->Capacity) {
            // Amortize growth without doubling a bucket's retained memory.
            NewCapacity = B->Capacity == 0 ? Need : Need + Need / 4;
        } else if (Need < B->Capacity / 2) {
            // A rare large policy must not permanently inflate the bucket.
            NewCapacity = Need + Need / 4;
        }
        NewBytes = allocationBytes(NewCapacity);
        if (NewBytes <= OldBytes || reserve(NewBytes - OldBytes)) {
            break;
        }
        // Under pressure, try an exact fit before discarding entries.
        NewCapacity = Need;
        NewBytes = allocationBytes(NewCapacity);
        if (NewBytes <= OldBytes || reserve(NewBytes - OldBytes)) {
            break;
        }
        const std::size_t Victim = leastRecent(B, Slot);
        if (Victim != NUM_WAYS) {
            erase(B, Victim);
            continue;
        }
        if (!reclaimAndReserve(NewBytes - OldBytes, BIdx)) {
            return false;
        }
        break;
    }

    const std::size_t Offset = offsetOf(B, Slot);
    const std::size_t OldSize = entryFloats(B->NumMoves[Slot]);
    const std::size_t Tail = B->Size - Offset - OldSize;
    std::unique_ptr<float[]> NewData;
    float* Data = B->Data.get();
    if (NewCapacity != B->Capacity) {
        NewData.reset(new (std::nothrow) float[NewCapacity]);
        if (NewData == nullptr) {
            if (NewBytes > OldBytes) {
                PayloadBytes.fetch_sub(NewBytes - OldBytes,
                                       std::memory_order_relaxed);
            }
            return false;
        }
        Data = NewData.get();
        if (Offset != 0) {
            std::memcpy(Data, B->Data.get(), Offset * sizeof(float));
        }
        if (Tail != 0) {
            std::memcpy(Data + Offset + NewSize,
                        B->Data.get() + Offset + OldSize,
                        Tail * sizeof(float));
        }
    } else if (Tail != 0 && NewSize != OldSize) {
        std::memmove(Data + Offset + NewSize, Data + Offset + OldSize,
                     Tail * sizeof(float));
    }
    Data[Offset] = WR;
    Data[Offset + 1] = D;
    std::memcpy(Data + Offset + 2, P, sizeof(float) * NumM);
    if (NewData != nullptr) {
        B->Data = std::move(NewData);
        B->Capacity = (uint32_t)NewCapacity;
        if (OldBytes > NewBytes) {
            PayloadBytes.fetch_sub(OldBytes - NewBytes,
                                   std::memory_order_relaxed);
        }
    }
    B->Size = (uint32_t)Need;
    B->NumMoves[Slot] = NumM;
    if (NewSize != OldSize) {
        updateOffsets(B);
    }
    B->Key[Slot].store(Hash, std::memory_order_relaxed);
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (W == Slot) {
            B->Recency[W].store(255, std::memory_order_relaxed);
        } else {
            const uint8_t R = B->Recency[W].load(std::memory_order_relaxed);
            B->Recency[W].store((uint8_t)(R >> 1), std::memory_order_relaxed);
        }
    }
    return true;
}

bool EvalCache::load(uint64_t Hash, EvalInfo* EI) {
    Bucket* B = &Buckets[bucketIndex(Hash)];
    if (B->isWriting()) {
        return false;
    }
    for (std::size_t W = 0; W < NUM_WAYS; ++W) {
        if (B->Key[W].load(std::memory_order_relaxed) != Hash) {
            continue;
        }
        std::shared_lock Guard(*B, std::try_to_lock);
        if (!Guard.owns_lock()) {
            return false;
        }
        const uint16_t NumM = B->NumMoves[W];
        if (NumM == 0 || B->Key[W].load(std::memory_order_relaxed) != Hash) {
            continue;
        }
        try {
            if (EI->Policy.size() < NumM) {
                EI->Policy.resize(NumM);
            }
        } catch (const std::bad_alloc&) {
            return false;
        }
        const float* Data = B->Data.get() + offsetOf(B, W);
        EI->NumMoves = NumM;
        EI->WinRate = Data[0];
        EI->DrawRate = Data[1];
        std::memcpy(EI->Policy.data(), Data + 2, sizeof(float) * NumM);
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
