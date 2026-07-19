//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_ALLOCATOR_SLAB_H
#define NSHOGI_ENGINE_ALLOCATOR_SLAB_H

#include "../lock/locktype.h"
#include "allocator.h"
#include "prefault.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __linux__

#include <sys/mman.h>

#endif

namespace nshogi {
namespace engine {
namespace allocator {

// Exact-size-class slab allocator, tailored to the Edge-array
// workload: every request is sizeof(Edge) (16 bytes) times the number
// of legal moves, so rounding sizes up to the next multiple of 16
// wastes nothing and gives one exact class per request size.
//
// The arena is carved into fixed-size slabs. A slab is bound to one
// class while it holds live objects and returns to a shared empty pool
// when the last object is freed, so memory moves freely between
// classes over time. Objects carry no header: all bookkeeping lives in
// a side table indexed by slab number, and a freed object stores the
// intra-slab free-list link in its own first bytes.
//
// Concurrency is sharded: each thread is assigned a home shard with
// its own lock and partial-slab lists, so threads working on
// different shards never contend (the same idea that lets glibc's
// per-thread arenas scale). A slab is owned by the shard that took it
// from the empty pool, and free() locks the owner's shard, so a
// cross-thread free only contends with that one shard. The shared
// empty pool is touched once per slab lifetime, not per object. Only
// when the pool is exhausted does malloc() fall back to serving from
// another shard's partial slab, so sharding costs no capacity at the
// point where capacity matters.
//
// Requests larger than MaxClassBytes (far above the largest possible
// Edge array) are not served and return nullptr.
template <lock::LockType LockT = std::mutex>
class SlabAllocator : public Allocator {
 public:
    SlabAllocator()
        : Size(0)
        , NumSlabs(0)
        , Memory(nullptr)
        , ShardCount(DefaultNumShards)
        , Shards(new Shard[DefaultNumShards])
        , EmptyHead(InvalidIndex) {
        initShards();
    }

    ~SlabAllocator() {
        if (Memory != nullptr) {
#ifdef __linux__
            munmap(Memory, Size);
#else
            std::free(Memory);
#endif
        }
    }

    void resize(std::size_t Size_) override {
        if (Memory != nullptr) {
#ifdef __linux__
            munmap(Memory, Size);
#else
            std::free(Memory);
#endif
        }

        NumSlabs = Size_ >> SlabSizeShift;
        Size = NumSlabs << SlabSizeShift;
#ifdef __linux__
        Memory = mmap(nullptr, Size, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        madvise(Memory, Size, MADV_HUGEPAGE);
        // Commit every page now (as MAP_POPULATE did), but on all
        // cores, so that no page-fault latency leaks into malloc()
        // during the search.
        prefaultRegionParallel(Memory, Size);
#else
        Memory = std::malloc(Size);
#endif

        Slabs.assign(NumSlabs, Slab{});
        initShards();

        // All slabs start in the shared empty pool.
        EmptyHead = InvalidIndex;
        for (std::size_t S = NumSlabs; S-- > 0;) {
            Slabs[S].Next = EmptyHead;
            EmptyHead = (uint32_t)S;
        }
    }

    void* malloc(std::size_t Size_) override {
        const std::size_t ClassIndex = getClassIndex(Size_);
        if (ClassIndex >= NumClasses) {
            return nullptr;
        }

        const std::size_t Stride = ClassIndex * Alignment;
        const uint32_t ObjectsPerSlab = (uint32_t)(SlabSize / Stride);
        // Popular small classes are sharded by thread for
        // scalability. Large classes hold few objects per slab, so
        // per-thread slabs would multiply half-empty slabs across
        // shards; they are rare, so binding them to a fixed shard
        // costs no scalability and keeps one partial set per class.
        const std::size_t Home = Stride >= ClassShardedMinStride
                                     ? ClassIndex % ShardCount
                                     : myThreadIndex() % ShardCount;

        {
            std::lock_guard<LockT> Lk(Shards[Home].Lock);

            uint32_t S = Shards[Home].PartialHead[ClassIndex];
            if (S == InvalidIndex) {
                // Take a slab from the shared empty pool and bind it
                // to this class, owned by the home shard.
                S = popEmptySlab();
                if (S != InvalidIndex) {
                    Slab& NewSlab = Slabs[S];
                    NewSlab.ClassIndex = (uint32_t)ClassIndex;
                    NewSlab.LiveCount = 0;
                    NewSlab.FreeHead = InvalidIndex;
                    NewSlab.BumpNext = 0;
                    NewSlab.Owner = (uint32_t)Home;
                    pushPartial(Home, ClassIndex, S);
                }
            }
            if (S != InvalidIndex) {
                return takeObject(Home, S, ClassIndex, Stride,
                                  ObjectsPerSlab);
            }
        }

        // Near-full slow path: the pool is exhausted and the home
        // shard has no partial slab of this class. Serve from any
        // shard that still has one, so sharding strands no capacity.
        // One lock is held at a time, so the scan cannot deadlock.
        for (std::size_t V = 0; V < ShardCount; ++V) {
            std::lock_guard<LockT> Lk(Shards[V].Lock);
            const uint32_t S = Shards[V].PartialHead[ClassIndex];
            if (S != InvalidIndex) {
                return takeObject(V, S, ClassIndex, Stride, ObjectsPerSlab);
            }
        }

        return nullptr;
    }

    void free(void* Ptr) override {
        if (Ptr == nullptr) {
            return;
        }

        const std::size_t Offset =
            (std::size_t)(static_cast<char*>(Ptr) -
                          static_cast<char*>(Memory));
        assert(Offset < Size);

        const uint32_t S = (uint32_t)(Offset >> SlabSizeShift);

        // The owner is stable without the lock: this object is still
        // live, so LiveCount >= 1, so the slab cannot return to the
        // pool and be rebound while we are here. The malloc() that
        // handed the object out ordered the Owner write before us via
        // the owner shard's lock.
        const std::size_t Owner = Slabs[S].Owner;

        std::lock_guard<LockT> Lk(Shards[Owner].Lock);

        Slab& Sl = Slabs[S];
        assert(Sl.Owner == Owner);

        const std::size_t ClassIndex = Sl.ClassIndex;
        const std::size_t Stride = ClassIndex * Alignment;
        const uint32_t ObjectsPerSlab = (uint32_t)(SlabSize / Stride);
        const uint32_t Object =
            (uint32_t)((Offset & (SlabSize - 1)) / Stride);

        assert(Sl.LiveCount > 0);

        const bool WasFull = (Sl.LiveCount == ObjectsPerSlab);

        *reinterpret_cast<uint32_t*>(Ptr) = Sl.FreeHead;
        Sl.FreeHead = Object;
        --Sl.LiveCount;

        Shards[Owner].Used.fetch_sub(Stride, std::memory_order_relaxed);

        if (WasFull) {
            pushPartial(Owner, ClassIndex, S);
        }

        if (Sl.LiveCount == 0) {
            // The slab is empty again: unbind it from the class and
            // return it to the shared pool.
            removePartial(Owner, ClassIndex, S);
            Sl.ClassIndex = 0;
            pushEmptySlab(S);
        }
    }

    std::size_t getTotal() const override {
        return Size;
    }

    std::size_t getUsed() const override {
        std::size_t Sum = 0;
        for (std::size_t W = 0; W < ShardCount; ++W) {
            Sum += Shards[W].Used.load(std::memory_order_relaxed);
        }
        return Sum;
    }

    // The number of shards should roughly match the number of
    // allocating threads (the search workers): fewer shards restores
    // lock contention, more shards disperses partial slabs for no
    // gain. Must be called while no allocation is live, like
    // resize().
    void setNumShards(std::size_t NumShards) {
        assert(getUsed() == 0);

        if (NumShards < 1) {
            NumShards = 1;
        }
        if (NumShards > MaxNumShards) {
            NumShards = MaxNumShards;
        }

        ShardCount = NumShards;
        Shards.reset(new Shard[ShardCount]);
        initShards();
    }

    std::size_t getNumShards() const {
        return ShardCount;
    }

    std::size_t getFree() const override {
        return getTotal() - getUsed();
    }

    bool isAllBlockFree() const {
        for (std::size_t S = 0; S < NumSlabs; ++S) {
            if (Slabs[S].LiveCount != 0) {
                return false;
            }
        }
        return true;
    }

 private:
    struct Slab {
        uint32_t ClassIndex = 0;
        uint32_t LiveCount = 0;
        uint32_t FreeHead = InvalidIndex;
        uint32_t BumpNext = 0;
        uint32_t Next = InvalidIndex;
        uint32_t Previous = InvalidIndex;
        uint32_t Owner = 0;
    };

    constexpr static std::size_t Alignment = 16;
    constexpr static std::size_t SlabSizeShift = 16; // 64 KiB.
    constexpr static std::size_t SlabSize = 1ULL << SlabSizeShift;
    // Largest servable request. The largest Edge array is 593 legal
    // moves x 16 bytes = 9488 bytes, well below this.
    constexpr static std::size_t MaxClassBytes = 16ULL * 1024;
    constexpr static std::size_t NumClasses =
        MaxClassBytes / Alignment + 1;
    constexpr static uint32_t InvalidIndex = 0xffffffffU;
    // Classes with at most SlabSize / ClassShardedMinStride (= 16)
    // objects per slab are class-sharded instead of thread-sharded.
    constexpr static std::size_t ClassShardedMinStride = 4096;
    constexpr static std::size_t DefaultNumShards = 8;
    constexpr static std::size_t MaxNumShards = 64;

    static_assert(MaxClassBytes * 4 <= SlabSize,
                  "every slab must hold at least four objects");

    // Per-shard state on its own cache lines: lock, partial-slab list
    // heads, and the used-bytes counter for slabs this shard owns.
    struct alignas(128) Shard {
        LockT Lock;
        std::atomic<uint64_t> Used;
        uint32_t PartialHead[NumClasses];
    };

    constexpr static std::size_t getClassIndex(std::size_t Size_) {
        return Size_ == 0 ? 1 : (Size_ + Alignment - 1) / Alignment;
    }

    // Threads are numbered on first use; the home shard is the
    // thread number modulo the runtime shard count, i.e. round-robin.
    static std::size_t myThreadIndex() {
        static std::atomic<uint32_t> Counter{0};
        thread_local static const uint32_t Index =
            Counter.fetch_add(1, std::memory_order_relaxed);
        return Index;
    }

    void initShards() {
        for (std::size_t W = 0; W < ShardCount; ++W) {
            for (std::size_t I = 0; I < NumClasses; ++I) {
                Shards[W].PartialHead[I] = InvalidIndex;
            }
            Shards[W].Used.store(0, std::memory_order_relaxed);
        }
    }

    char* objectAt(uint32_t S, uint32_t Object, std::size_t Stride) const {
        return static_cast<char*>(Memory) + ((std::size_t)S << SlabSizeShift) +
               (std::size_t)Object * Stride;
    }

    // Pop one object from slab S. The caller holds the lock of the
    // shard that owns S, and S has a free object.
    void* takeObject(std::size_t ShardIndex, uint32_t S,
                     std::size_t ClassIndex, std::size_t Stride,
                     uint32_t ObjectsPerSlab) {
        Slab& Sl = Slabs[S];
        assert(Sl.ClassIndex == ClassIndex);
        assert(Sl.Owner == ShardIndex);

        uint32_t Object;
        if (Sl.FreeHead != InvalidIndex) {
            // Reuse a freed object; it stores the next link in its
            // first bytes.
            Object = Sl.FreeHead;
            Sl.FreeHead = *reinterpret_cast<const uint32_t*>(
                objectAt(S, Object, Stride));
        } else {
            Object = Sl.BumpNext;
            ++Sl.BumpNext;
        }

        ++Sl.LiveCount;
        if (Sl.LiveCount == ObjectsPerSlab) {
            // A full slab leaves the partial list; free() finds it
            // again purely by the pointer's slab number.
            removePartial(ShardIndex, ClassIndex, S);
        }

        Shards[ShardIndex].Used.fetch_add(Stride,
                                          std::memory_order_relaxed);

        void* Ptr = objectAt(S, Object, Stride);
        assert(reinterpret_cast<uint64_t>(Ptr) % Alignment == 0);
        return Ptr;
    }

    // The empty pool has its own lock, always taken while holding a
    // shard lock (shard -> pool order everywhere, so no deadlock).
    uint32_t popEmptySlab() {
        std::lock_guard<LockT> Lk(EmptyLock);
        const uint32_t S = EmptyHead;
        if (S != InvalidIndex) {
            EmptyHead = Slabs[S].Next;
        }
        return S;
    }

    void pushEmptySlab(uint32_t S) {
        std::lock_guard<LockT> Lk(EmptyLock);
        Slabs[S].Next = EmptyHead;
        EmptyHead = S;
    }

    void pushPartial(std::size_t ShardIndex, std::size_t ClassIndex,
                     uint32_t S) {
        Slab& Sl = Slabs[S];
        Sl.Previous = InvalidIndex;
        Sl.Next = Shards[ShardIndex].PartialHead[ClassIndex];
        if (Sl.Next != InvalidIndex) {
            Slabs[Sl.Next].Previous = S;
        }
        Shards[ShardIndex].PartialHead[ClassIndex] = S;
    }

    void removePartial(std::size_t ShardIndex, std::size_t ClassIndex,
                       uint32_t S) {
        Slab& Sl = Slabs[S];
        if (Sl.Previous != InvalidIndex) {
            Slabs[Sl.Previous].Next = Sl.Next;
        } else {
            assert(Shards[ShardIndex].PartialHead[ClassIndex] == S);
            Shards[ShardIndex].PartialHead[ClassIndex] = Sl.Next;
        }
        if (Sl.Next != InvalidIndex) {
            Slabs[Sl.Next].Previous = Sl.Previous;
        }
        Sl.Next = InvalidIndex;
        Sl.Previous = InvalidIndex;
    }

    std::size_t Size;
    std::size_t NumSlabs;

    void* Memory;

    std::vector<Slab> Slabs;
    std::size_t ShardCount;
    std::unique_ptr<Shard[]> Shards;

    LockT EmptyLock;
    uint32_t EmptyHead;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_SLAB_H
