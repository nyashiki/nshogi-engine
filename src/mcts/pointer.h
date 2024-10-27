#ifndef NSHOGI_ENGINE_MCTS_POINTER_H
#define NSHOGI_ENGINE_MCTS_POINTER_H

#include "../allocator/allocator.h"

#include <cassert>
#include <cinttypes>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename T>
class Pointer {
 public:
    Pointer()
        : P(nullptr) {
    }

    Pointer(Pointer&& Other) noexcept
        : P(Other.P)
        , S(Other.S) {
        Other.P = nullptr;
    }

    ~Pointer() {
        // destroy() must be called before the destructor is called.
        assert(P == nullptr);
    }

    Pointer& operator=(Pointer&& Other) noexcept {
        if (this != &Other) {
            assert(P == nullptr);

            P = Other.P;
            S = Other.S;

            Other.P = nullptr;
        }

        return *this;
    }

    T* get() {
        return P;
    }

    const T* get() const {
        return P;
    }

    T& operator[](std::size_t Index) {
        return P[Index];
    }

    const T& operator[](std::size_t Index) const {
        return P[Index];
    }

    bool operator==(T* Ptr) const {
        return P == Ptr;
    }

    bool operator!=(T* Ptr) const {
        return P != Ptr;
    }

    T* operator->() {
        return P;
    }

    const T* operator->() const {
        return P;
    }

    template <typename... ArgTypes>
    T* malloc(allocator::Allocator* Allocator, ArgTypes... Args) {
        P = static_cast<T*>(Allocator->malloc(sizeof(T)));

        // Failed to allocate memory.
        if (P == nullptr) {
            return P;
        }

        new (P) T(Args...);
        S = 1;

        return P;
    }

    T* mallocArray(allocator::Allocator* Allocator, std::size_t Size) {
        P = static_cast<T*>(Allocator->malloc(sizeof(T) * Size));

        // Failed to allocate memory.
        if (P == nullptr) {
            return P;
        }

        for (std::size_t I = 0; I < Size; ++I) {
            new (&P[I]) T();
        }
        S = Size;

        return P;
    }

    void destroy(allocator::Allocator* Allocator) {
        if (P != nullptr) {
            for (std::size_t I = 0; I < S; ++I) {
                P[I].~T();
            }
            Allocator->free(P);
            P = nullptr;
        }
    }

 private:
    T* P;
    std::size_t S;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_POINTER_H
