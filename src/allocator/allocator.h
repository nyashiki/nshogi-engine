#ifndef NSHOGI_ENGINE_ALLOCATOR_ALLOCATOR_H
#define NSHOGI_ENGINE_ALLOCATOR_ALLOCATOR_H

#include <cinttypes>

namespace nshogi {
namespace engine {
namespace allocator {

class Allocator {
 public:
    virtual ~Allocator() {
    }

    // Before calling this function, one must have responsibility
    // to have called free() for all memory allocated by calling malloc().
    virtual void resize(std::size_t Size) = 0;
    virtual void* malloc(std::size_t Size) = 0;
    virtual void free(void* Mem) = 0;

    virtual std::size_t getTotal() const = 0;
    virtual std::size_t getUsed() const = 0;
    virtual std::size_t getFree() const = 0;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_ALLOCATOR_H
