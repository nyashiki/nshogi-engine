#ifndef NSHOGI_ENGINE_ALLOCATOR_SEGREGATED_FREE_LIST_H
#define NSHOGI_ENGINE_ALLOCATOR_SEGREGATED_FREE_LIST_H

#include "allocator.h"
#include "../lock/spinlock.h"

#include <atomic>
#include <mutex>

#include <cassert>
#include <cstring>

#include <iostream>
#include <sys/mman.h>

namespace nshogi {
namespace engine {
namespace allocator {

class SegregatedFreeListAllocator : public Allocator {
 public:
    SegregatedFreeListAllocator()
        : Size(0)
        , Used(0)
        , Memory(nullptr)
        , AvailableListBits(0) {
    }

    ~SegregatedFreeListAllocator() {
        if (Memory != nullptr) {
            munmap(Memory, Size);
        }
    }

    void resize(std::size_t Size_) override {
        if (Memory != nullptr) {
            munmap(Memory, Size);
        }

        const std::size_t AlignedSize = getAlignedSize(Size_);
        Size = sizeof(Header1) + AlignedSize + Alignment + sizeof(Header1);
        Memory = mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0);

        AlignedMemory = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(Memory) + sizeof(Header1) + AlignmentMask) & ~AlignmentMask);

        for (std::size_t I = 0; I < 64; ++I) {
            FreeLists[I] = nullptr;
        }

        Header2* Header = reinterpret_cast<Header2*>(AlignedMemory);
        Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(AlignedMemory) - sizeof(Header1));
        H1->setSize(AlignedSize);
        Footer* Foot = reinterpret_cast<Footer*>(reinterpret_cast<char*>(H1) + AlignedSize - sizeof(Footer));
        Foot->Header = Header;
        addToFreeList(Header);

        Header1* LastH1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Foot) + sizeof(Footer));
        LastH1->setSize(0);
        LastH1->setUsed();
    }

    void* malloc(std::size_t Size_) override {
        const std::size_t AlignedSize = getAlignedSize(Size_ + sizeof(Footer) + sizeof(Header1));
        const std::size_t MinimumIndex = (std::size_t)(64 - __builtin_clzll(AlignedSize - 1));

        std::lock_guard<lock::SpinLock> Lk(SpinLock);

        const std::size_t TargetBits = AvailableListBits & (0xffffffffffffffffULL << MinimumIndex);

        if (TargetBits == 0) {
            return nullptr;
        }

        const std::size_t Index = (std::size_t)__builtin_ctzll(TargetBits);
        assert(FreeLists[Index] != nullptr);

        // Pop out the first block.
        Header2* Header = FreeLists[Index];
        FreeLists[Index] = FreeLists[Index]->Next;
        if (FreeLists[Index] == nullptr) {
            AvailableListBits ^= (1ULL << Index);
        } else {
            FreeLists[Index]->Previous = nullptr;
        }

        Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Header) - sizeof(Header1));

        // Split the block if the remaining size is enough.
        const std::size_t RemainingSize = H1->getSize() - AlignedSize;
        assert(RemainingSize % Alignment == 0);

        if (RemainingSize > sizeof(Footer) + sizeof(Header1)) {
            H1->setSize(AlignedSize);
            Footer* Foot = reinterpret_cast<Footer*>(reinterpret_cast<char*>(Header) +
                    AlignedSize - sizeof(Header1) - sizeof(Footer));

            Foot->Header = Header;

            Header2* NewHeader = reinterpret_cast<Header2*>(reinterpret_cast<char*>(Header) + AlignedSize);
            Header1* NewH1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(NewHeader) - sizeof(Header1));
            Footer* NewFooter = reinterpret_cast<Footer*>(reinterpret_cast<char*>(NewHeader) + RemainingSize - sizeof(Header1) - sizeof(Footer));

            NewH1->setSize(RemainingSize);
            NewFooter->Header = NewHeader;
            addToFreeList(NewHeader);
        }

        H1->setUsed();

        Used.fetch_add(H1->getSize(), std::memory_order_relaxed);

        assert(reinterpret_cast<uint64_t>(Header) % Alignment == 0);
        return Header;
    }

    void free(void* Ptr) override {
        if (Ptr == nullptr) {
            return;
        }

        Header2* Header = reinterpret_cast<Header2*>(Ptr);
        Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Header) - sizeof(Header1));
        Used.fetch_sub(H1->getSize(), std::memory_order_relaxed);

        std::lock_guard<lock::SpinLock> Lk(SpinLock);

        H1->toggleUsed();
        assert(!H1->getIsUsed());

        if (Ptr != AlignedMemory) {
            Footer* PreviousFooter = reinterpret_cast<Footer*>(reinterpret_cast<char*>(H1) - sizeof(Footer));
            Header2* PreviousHeader = PreviousFooter->Header;
            Header1* PreviousH1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(PreviousHeader) - sizeof(Header1));

            if (!PreviousH1->getIsUsed()) {
                removeFromFreeList(PreviousHeader);

                PreviousH1->setSize(PreviousH1->getSize() + H1->getSize());

                Footer* CurrentFotter = reinterpret_cast<Footer*>(reinterpret_cast<char*>(Header) + H1->getSize() - sizeof(Header1) - sizeof(Footer));
                CurrentFotter->Header = PreviousHeader;

                H1 = PreviousH1;
                Header = PreviousHeader;
            }
        }

        Header2* NextHeader = reinterpret_cast<Header2*>(reinterpret_cast<char*>(Header) + H1->getSize());
        Header1* NextH1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(NextHeader) - sizeof(Header1));

        if (!NextH1->getIsUsed()) {
            removeFromFreeList(NextHeader);

            H1->setSize(H1->getSize() + NextH1->getSize());

            Footer* NextFooter = reinterpret_cast<Footer*>(reinterpret_cast<char*>(NextHeader) + NextH1->getSize() - sizeof(Header1) - sizeof(Footer));
            NextFooter->Header = Header;
        }

        addToFreeList(Header);
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

    void printFragmentation() const {
        for (std::size_t I = 0; I < 64; I += 8) {
            for (std::size_t J = I; J < I + 8; ++J) {
                std::size_t Count = 0;
                Header2* Header = FreeLists[J];
                while (Header != nullptr) {
                    Header = Header->Next;
                    ++Count;
                }
                if (Count > 0) {
                    printf("\x1b[31m%02zu: %10zu\x1b[39m, ", J, Count);
                } else {
                    printf("%02zu: %10zu, ", J, Count);
                }
            }
            printf("\n");
        }
    }

    bool isAllBlockFree() const {
        Header2* Header = reinterpret_cast<Header2*>(AlignedMemory);

        while (true) {
            Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Header) - sizeof(Header1));

            if (H1->getSize() == 0) {
                break;
            }

            if (H1->getIsUsed()) {
                return false;
            }

            Header = reinterpret_cast<Header2*>(reinterpret_cast<char*>(Header) + H1->getSize());
        }

        return true;
    }

 private:
    struct Header1 {
     public:
        std::size_t getSize() const {
            return (std::size_t)(Data & ~0b11ULL);
        }

        bool getIsUsed() const {
            return (bool)(Data & 0b1);
        }

        void toggleUsed() {
            Data ^= 0b1;
        }

        void setUsed() {
            Data |= 0b1;
        }

        void setSize(std::size_t Size_) {
            Data = (uint64_t)(Size_);
        }

     private:
        uint64_t Data;
    };

    struct Header2 {
        Header2* Next;
        Header2* Previous;
    };

    struct Footer {
        Header2* Header;
    };

    constexpr static std::size_t Alignment = 16;
    constexpr static std::size_t AlignmentMask = Alignment - 1;

    constexpr static std::size_t getAlignedSize(std::size_t Size_) {
        return (Size_ + AlignmentMask) & ~AlignmentMask;
    }

    void addToFreeList(Header2* Header, std::size_t Index) {
        if (FreeLists[Index] != nullptr) {
            FreeLists[Index]->Previous = Header;
        }

        Header->Next = FreeLists[Index];
        Header->Previous = nullptr;
        FreeLists[Index] = Header;

        AvailableListBits |= (1ULL << Index);
    }

    void addToFreeList(Header2* Header) {
        Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Header) - sizeof(Header1));
        std::size_t Index = (std::size_t)(63 - __builtin_clzll(H1->getSize()));
        addToFreeList(Header, Index);
    }

    void removeFromFreeList(Header2* Header) {
        if (Header->Previous != nullptr) { // This block is not the first element.
            // FreeList[Index]:
            //     Old: ... [ (a) ][ This block. ][ (b) ] ...
            //     New: ... [ (a) ][ (b) ] ...
            Header->Previous->Next = Header->Next; // (a) ==> (b).
            if (Header->Next != nullptr) {
                Header->Next->Previous = Header->Previous;  // (a) <== (b).
            }
        } else { // This block is the first element.
            // FreeList[Index]:
            //     Old: [ This block. ][ (a) ] ...
            //     New: [ (a) ] ...

            const Header1* H1 = reinterpret_cast<Header1*>(reinterpret_cast<char*>(Header) - sizeof(Header1));
            const std::size_t Index = (std::size_t)(63 - __builtin_clzll(H1->getSize()));
            assert(FreeLists[Index] == Header);

            FreeLists[Index] = Header->Next;

            if (FreeLists[Index] == nullptr) {
                assert((AvailableListBits & (1ULL << Index)) != 0);
                AvailableListBits ^= (1ULL << Index);
            } else {
                FreeLists[Index]->Previous = nullptr;
            }
        }
    }

    std::size_t Size;
    std::atomic<uint64_t> Used;

    void* Memory;
    void* AlignedMemory;

    Header2* FreeLists[64];
    uint64_t AvailableListBits;

    lock::SpinLock SpinLock;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_SEGREGATED_FREE_LIST_H
