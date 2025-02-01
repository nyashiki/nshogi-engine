//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_ALLOCATOR_DEFAULT_H
#define NSHOGI_ENGINE_ALLOCATOR_DEFAULT_H

#include "allocator.h"

namespace nshogi {
namespace engine {
namespace allocator {

class DefaultAllocator : public Allocator {
 public:
    DefaultAllocator();
    ~DefaultAllocator() override;

    void resize(std::size_t Size) override;
    void* malloc(std::size_t Size) override;
    void free(void* Mem) override;

    std::size_t getTotal() const override;
    std::size_t getUsed() const override;
    std::size_t getFree() const override;
};

} // namespace allocator
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_ALLOCATOR_DEFAULT_H
