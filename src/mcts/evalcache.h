//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
#define NSHOGI_ENGINE_MCTS_EVALCACHE_H

#include <cinttypes>
#include <cstring>
#include <memory>
#include <mutex>

#include <nshogi/core/state.h>

namespace nshogi {
namespace engine {
namespace mcts {

class EvalCache {
 public:
    static constexpr std::size_t MAX_CACHE_MOVES_COUNT = 164;

    struct EvalInfo {
        uint16_t NumMoves;
        float Policy[MAX_CACHE_MOVES_COUNT];
        float WinRate;
        float DrawRate;

        EvalInfo& operator=(const EvalInfo& EI) {
            std::memcpy(this, &EI, sizeof(EvalInfo));
            return *this;
        }
    };

    EvalCache(std::size_t MemorySize);

    bool store(uint64_t Hash, uint16_t NumM, const float* P, float WR, float D);
    bool load(const core::State&, EvalInfo*);

 private:
    static constexpr std::size_t CACHE_BUNDLE_SIZE = 3;

    struct CacheData {
        bool IsUsed;
        uint64_t Hash64 = 0;
        EvalInfo EInfo;
        CacheData* Next;
        CacheData* Prev;
    };

    struct CacheBundle {
        std::mutex Mtx;
        CacheData* Head;
    };

    const std::size_t NumBundle;
    const std::unique_ptr<CacheData[]> Memory;
    std::unique_ptr<CacheBundle[]> CacheStorage;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_EVALCACHE_H
