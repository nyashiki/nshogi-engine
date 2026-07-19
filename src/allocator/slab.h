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
// Requests larger than MaxClassBytes (far above the largest possible
// Edge array) are not served and return nullptr.
template <lock::LockType LockT = std::mutex>
class SlabAllocator : public Allocator {
 public:
    SlabAllocator()
        : Size(0)
        , NumSlabs(0)
        , Used(0)
        , Memory(nullptr)
        , EmptyHead(InvalidIndex) {
        for (std::size_t I = 0; I < NumClasses; ++I) {
            PartialHead[I] = InvalidIndex;
        }
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
        for (std::size_t I = 0; I < NumClasses; ++I) {
            PartialHead[I] = InvalidIndex;
        }
        Used.store(0, std::memory_order_relaxed);

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

        std::lock_guard<LockT> Lk(Lock);

        uint32_t S = PartialHead[ClassIndex];
        if (S == InvalidIndex) {
            // Take a slab from the shared empty pool and bind it to
            // this class.
            S = EmptyHead;
            if (S == InvalidIndex) {
                return nullptr;
            }
            EmptyHead = Slabs[S].Next;

            Slab& NewSlab = Slabs[S];
            NewSlab.ClassIndex = (uint32_t)ClassIndex;
            NewSlab.LiveCount = 0;
            NewSlab.FreeHead = InvalidIndex;
            NewSlab.BumpNext = 0;
            pushPartial(ClassIndex, S);
        }

        Slab& Sl = Slabs[S];
        assert(Sl.ClassIndex == ClassIndex);

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
            removePartial(ClassIndex, S);
        }

        Used.fetch_add(Stride, std::memory_order_relaxed);

        void* Ptr = objectAt(S, Object, Stride);
        assert(reinterpret_cast<uint64_t>(Ptr) % Alignment == 0);
        return Ptr;
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

        std::lock_guard<LockT> Lk(Lock);

        Slab& Sl = Slabs[S];
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

        Used.fetch_sub(Stride, std::memory_order_relaxed);

        if (WasFull) {
            pushPartial(ClassIndex, S);
        }

        if (Sl.LiveCount == 0) {
            // The slab is empty again: unbind it from the class and
            // return it to the shared pool.
            removePartial(ClassIndex, S);
            Sl.ClassIndex = 0;
            Sl.Next = EmptyHead;
            EmptyHead = S;
        }
    }

    std::size_t getTotal() const override {
        return Size;
    }

    std::size_t getUsed() const override {
        return Used.load(std::memory_order_relaxed);
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

    static_assert(MaxClassBytes * 4 <= SlabSize,
                  "every slab must hold at least four objects");

    constexpr static std::size_t getClassIndex(std::size_t Size_) {
        return Size_ == 0 ? 1 : (Size_ + Alignment - 1) / Alignment;
    }

    char* objectAt(uint32_t S, uint32_t Object, std::size_t Stride) const {
        return static_cast<char*>(Memory) + ((std::size_t)S << SlabSizeShift) +
               (std::size_t)Object * Stride;
    }

    void pushPartial(std::size_t ClassIndex, uint32_t S) {
        Slab& Sl = Slabs[S];
        Sl.Previous = InvalidIndex;
        Sl.Next = PartialHead[ClassIndex];
        if (Sl.Next != InvalidIndex) {
            Slabs[Sl.Next].Previous = S;
        }
        PartialHead[ClassIndex] = S;
    }

    void removePartial(std::size_t ClassIndex, uint32_t S) {
        Slab& Sl = Slabs[S];
        if (Sl.Previous != InvalidIndex) {
            Slabs[Sl.Previous].Next = Sl.Next;
        } else {
            assert(PartialHead[ClassIndex] == S);
            PartialHead[ClassIndex] = Sl.Next;
        }
        if (Sl.Next != InvalidIndex) {
            Slabs[Sl.Next].Previous = Sl.Previous;
        }
        Sl.Next = InvalidIndex;
        Sl.Previous = InvalidIndex;
    }

    std::size_t Size;
    std::size_t NumSlabs;
    std::atomic<uint64_t> Used;

    void* Memory;

    std::vector<Slab> Slabs;
    uint32_t PartialHead[NumClasses];
    uint32_t EmptyHead;

    LockT Lock;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_SLAB_H
