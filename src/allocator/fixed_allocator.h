#ifndef NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H
#define NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H

#include "allocator.h"
#include "../lock/spinlock.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

namespace nshogi {
namespace engine {
namespace allocator {

template <std::size_t FixedSize>
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
            munmap(Memory, Size);
            Memory = nullptr;
        }
    }

    void resize(std::size_t Size_) override {
        if (Memory != nullptr) {
            FreeList = nullptr;
            munmap(Memory, Size);
        }

        Size = Size_;
        Used = 0;
        Memory = mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0);
        std::memset(Memory, 0, Size);

        AlignedMemory = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(Memory) + AlignmentMask) & ~AlignmentMask);
        // for (AlignedMemory = reinterpret_cast<char*>(Memory); ; AlignedMemory = reinterpret_cast<char*>(AlignedMemory) + 1) {
        //     if (reinterpret_cast<uint64_t>(AlignedMemory) % Alignment == 0) {
        //         break;
        //     }
        // }

        FreeList = nullptr;
        Header* Previous = nullptr;

        for (void* Mem = AlignedMemory;
                (reinterpret_cast<char*>(Mem) + BlockSize) < reinterpret_cast<char*>(Memory) + Size;
                Mem = reinterpret_cast<char*>(Mem) + BlockSize) {
            Header* H = reinterpret_cast<Header*>(Mem);

            if (Previous != nullptr) {
                Previous->Next = H;
            } else {
                FreeList = H;
            }

            Previous = H;
        }

        if (Previous != nullptr) {
            Previous->Next = nullptr;
        }
    }

    void* malloc(std::size_t) override {
        if (FreeList == nullptr) {
            return nullptr;
        }

        Used.fetch_add(BlockSize, std::memory_order_relaxed);

        Header* Head;

        {
            std::lock_guard<lock::SpinLock> Lk(SpinLock);

            if (FreeList == nullptr) {
                return nullptr;
            }

            Head = FreeList;
            FreeList = FreeList->Next;
        }

        assert(reinterpret_cast<uint64_t>(Head) % Alignment == 0);
        return Head;
    }

    void free(void* Mem) override {
        Header* Block = static_cast<Header*>(Mem);

        {
            std::lock_guard<lock::SpinLock> Lk(SpinLock);
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

    const std::size_t Alignment = 16;
    const std::size_t AlignmentMask = Alignment - 1;
    const std::size_t BlockSize = (FixedSize + AlignmentMask) & ~AlignmentMask;

    std::size_t Size;
    std::atomic<std::size_t> Used;

    void* Memory;
    void* AlignedMemory;

    Header* FreeList;
    lock::SpinLock SpinLock;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_FIXED_ALLOCATOR_H
