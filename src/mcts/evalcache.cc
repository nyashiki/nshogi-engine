#include "evalcache.h"

namespace nshogi {
namespace engine {
namespace mcts {

EvalCache::EvalCache(std::size_t MemorySize)
    : NumBundle(MemorySize * 1024LL * 1024LL / (sizeof(CacheData) * CACHE_BUNDLE_SIZE))
    , Memory(std::make_unique<CacheData[]>(NumBundle * CACHE_BUNDLE_SIZE)) {

    CacheStorage = std::make_unique<CacheBundle[]>(NumBundle);

    CacheData* P = Memory.get();
    for (std::size_t I = 0; I < NumBundle; ++I) {
        CacheStorage[I].Head = P;
        CacheData* CacheElem = CacheStorage[I].Head;
        for (std::size_t J = 0; J < CACHE_BUNDLE_SIZE; ++J) {
            CacheElem->IsUsed = false;

            if (J == 0) {
                CacheElem->Prev = nullptr;
            } else {
                CacheElem->Prev = P - 1;
                CacheElem->Prev->Next = CacheElem;
            }

            if (J + 1 < CACHE_BUNDLE_SIZE) {
                CacheElem->Next = P + 1;
            } else {
                CacheElem->Next = nullptr;
            }

            CacheElem = CacheElem->Next;
            ++P;
        }
    }
}

bool EvalCache::store(uint64_t Hash, uint16_t NumM, const float* P, float WR, float D) {
    if (NumM > MAX_CACHE_MOVES_COUNT) {
        return false;
    }

    CacheBundle* Bundle = &CacheStorage[Hash % NumBundle];

    bool LockAcquired = Bundle->Mtx.try_lock();

    if (!LockAcquired) {
        return false;
    }

    CacheData* CacheElem = Bundle->Head;

    while (true) {
        if (!CacheElem->IsUsed) {
            break;
        }

        if (CacheElem->Hash64 == Hash && CacheElem->EInfo.NumMoves == NumM) {
            // The entry already exists.

            // Reorder.
            if (CacheElem->Prev != nullptr) {
                CacheElem->Prev->Next = CacheElem->Next;

                if (CacheElem->Next != nullptr) {
                    CacheElem->Next->Prev = CacheElem->Prev;
                }

                CacheElem->Next = Bundle->Head;
                CacheElem->Prev = nullptr;
                Bundle->Head = CacheElem;
            }

            Bundle->Mtx.unlock();
            return true;
        }

        if (CacheElem->Next == nullptr) {
            break;
        }

        CacheElem = CacheElem->Next;
    }

    // Reorder.
    if (CacheElem->Prev != nullptr) {
        CacheElem->Prev->Next = CacheElem->Next;

        if (CacheElem->Next != nullptr) {
            CacheElem->Next->Prev = CacheElem->Prev;
        }

        CacheElem->Next = Bundle->Head;
        CacheElem->Prev = nullptr;
        Bundle->Head = CacheElem;
    }

    CacheElem->IsUsed = true;
    CacheElem->Hash64 = Hash;

    CacheElem->EInfo.NumMoves = NumM;
    std::memcpy(CacheElem->EInfo.Policy, P, sizeof(float) * NumM);
    CacheElem->EInfo.WinRate = WR;
    CacheElem->EInfo.DrawRate = D;

    Bundle->Mtx.unlock();
    return true;
};

bool EvalCache::load(const core::State& St, EvalInfo* EI) {
    const uint64_t Hash = St.getHash();

    CacheBundle* Bundle = &CacheStorage[Hash % NumBundle];
    const bool LockAcquired = Bundle->Mtx.try_lock();
    if (!LockAcquired) {
        return false;
    }

    CacheData* CacheElem = Bundle->Head;

    while (CacheElem != nullptr) {
        if (!CacheElem->IsUsed) {
            break;
        }

        if (CacheElem->Hash64 == Hash) {
            *EI = CacheElem->EInfo;

            // Reorder.
            // If the found element is not located at the first position,
            // reorder so that it comes to the first position.
            if (CacheElem->Prev != nullptr) {
                CacheElem->Prev->Next = CacheElem->Next;

                if (CacheElem->Next != nullptr) {
                    CacheElem->Next->Prev = CacheElem->Prev;
                }

                CacheElem->Next = Bundle->Head;
                CacheElem->Prev = nullptr;
                Bundle->Head = CacheElem;
            }

            Bundle->Mtx.unlock();
            return true;
        }

        CacheElem = CacheElem->Next;
    }

    Bundle->Mtx.unlock();
    return false;
}

} // namespace mcts
} // namescape engine
} // namespace nshogi
