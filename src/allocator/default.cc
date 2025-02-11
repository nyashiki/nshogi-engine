//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "default.h"

#include <cstdlib>

namespace nshogi {
namespace engine {
namespace allocator {

DefaultAllocator::DefaultAllocator() {
}

DefaultAllocator::~DefaultAllocator() {
}

void DefaultAllocator::resize([[maybe_unused]] std::size_t Size) {
}

void* DefaultAllocator::malloc(std::size_t Size) {
    void* Ptr = std::malloc(Size);
    return Ptr;
}

void DefaultAllocator::free(void* Mem) {
    std::free(Mem);
}

std::size_t DefaultAllocator::getTotal() const {
    return 0;
}

std::size_t DefaultAllocator::getUsed() const {
    return 0;
}

std::size_t DefaultAllocator::getFree() const {
    return 0;
}

} // namespace allocator
} // namespace engine
} // namespace nshogi
