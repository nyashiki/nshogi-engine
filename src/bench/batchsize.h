//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_BENCH_BATCHSIZE_H
#define NSHOGI_ENGINE_BENCH_BATCHSIZE_H

#include <cstdint>

namespace nshogi {
namespace engine {
namespace bench {

void benchBatchSize(const char* Path, std::size_t Repeat);

} // namespace bench
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_BENCH_BATCHSIZE_H
