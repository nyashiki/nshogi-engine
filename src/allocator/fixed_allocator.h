//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H
#define NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H

#include "../lock/locktype.h"
#include "allocator.h"
#include "prefault.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <mutex>

#ifdef __linux__

#include <sys/mman.h>

#endif

namespace nshogi {
namespace engine {
namespace allocator {

template <std::size_t FixedSize, lock::LockType LockT = std::mutex>
class FixedAllocator : public Allocator {
 public:
    FixedAllocator()
        : Size(0)
        , Used(0)
        , Memory(nullptr)
        , FreeList(nullptr) {
    }

    ~FixedAllocator() override {
        FreeList = nullptr;

        if (Memory != nullptr) {
#ifdef __linux__
            munmap(Memory, Size);
#else
            std::free(Memory);
#endif
            Memory = nullptr;
        }
    }

    void resize(std::size_t Size_) override {
        if (Memory != nullptr) {
            FreeList = nullptr;
#ifdef __linux__
            munmap(Memory, Size);
#else
            std::free(Memory);
#endif
        }

        Size = Size_;
        Used = 0;
#ifdef __linux__
        // No MAP_POPULATE here: building the free list below writes
        // into every block and thus commits every page anyway, and it
        // does so on all cores instead of the kernel's sequential
        // populate.
        Memory = mmap(nullptr, Size, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        madvise(Memory, Size, MADV_HUGEPAGE);
#else
        Memory = std::malloc(Size);
#endif

        AlignedMemory = reinterpret_cast<void*>(
            (reinterpret_cast<std::size_t>(Memory) + AlignmentMask) &
            ~AlignmentMask);

        const std::size_t Usable =
            Size - (std::size_t)(reinterpret_cast<char*>(AlignedMemory) -
                                 reinterpret_cast<char*>(Memory));
        const std::size_t NumBlocks = Usable / BlockSize;

        // Build the same address-ordered free list as a sequential
        // construction would: block I points to block I + 1. Each
        // slice's last block points at the next slice's first block,
        // so the slices can be linked independently.
        forEachSliceParallel(NumBlocks, (32ULL << 20) / BlockSize,
                             [&](std::size_t Begin, std::size_t End) {
                                 for (std::size_t I = Begin; I < End; ++I) {
                                     blockAt(I)->Next = (I + 1 < NumBlocks)
                                                            ? blockAt(I + 1)
                                                            : nullptr;
                                 }
                             });

        FreeList = NumBlocks > 0 ? blockAt(0) : nullptr;
    }

    void* malloc(std::size_t) override {
        std::lock_guard<LockT> Lk(Lock);

        if (FreeList == nullptr) {
            return nullptr;
        }

        Used.fetch_add(BlockSize, std::memory_order_relaxed);

        Header* Head = FreeList;
        FreeList = FreeList->Next;

        assert(reinterpret_cast<uint64_t>(Head) % Alignment == 0);
        return Head;
    }

    void free(void* Mem) override {
        Header* Block = static_cast<Header*>(Mem);

        {
            std::lock_guard<LockT> Lk(Lock);
            Block->Next = FreeList;
            FreeList = Block;
        }

        Used.fetch_sub(BlockSize, std::memory_order_relaxed);
    }

    std::size_t getTotal() const override {
        return Size;
    }

    std::size_t getUsed() const override {
        return Used;
    }

    std::size_t getFree() const override {
        return getTotal() - getUsed();
    }

 private:
    struct Header {
        Header* Next;
    };

    Header* blockAt(std::size_t I) const {
        return reinterpret_cast<Header*>(
            reinterpret_cast<char*>(AlignedMemory) + I * BlockSize);
    }

    const std::size_t Alignment = 16;
    const std::size_t AlignmentMask = Alignment - 1;
    const std::size_t BlockSize = (FixedSize + AlignmentMask) & ~AlignmentMask;

    std::size_t Size;
    std::atomic<std::size_t> Used;

    void* Memory;
    void* AlignedMemory;

    Header* FreeList;
    LockT Lock;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H
