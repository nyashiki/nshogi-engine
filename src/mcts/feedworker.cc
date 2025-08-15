//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "feedworker.h"

#include "../math/math.h"
#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace mcts {

FeedWorker::FeedWorker(const Context* C, FeedQueue* FQueue, EvalCache* EC)
    : worker::Worker(true)
    , PContext(C)
    , Queue(FQueue)
    , ECache(EC) {

    spawnThread();
}

bool FeedWorker::doTask() {
    std::unique_ptr<Batch>&& B = Queue->get();

    if (B == nullptr) {
        return false;
    }

    feedResults(std::move(B));

    return true;
}

void FeedWorker::feedResults(std::unique_ptr<Batch>&& B) {
    if (PContext->isNaNFallbackEnabled()) {
        for (std::size_t I = 0; I < B->size(); ++I) {
            feedResult<true>(
                B->color(I),
                B->node(I),
                B->policy(I),
                B->winRate(I),
                B->drawRate(I),
                B->hash(I)
            );
        }
    } else {
        for (std::size_t I = 0; I < B->size(); ++I) {
            feedResult<false>(
                B->color(I),
                B->node(I),
                B->policy(I),
                B->winRate(I),
                B->drawRate(I),
                B->hash(I)
            );
        }
    }
}

template <bool NaNFallbackEnabled>
void FeedWorker::feedResult(
    core::Color SideToMove,
    Node* N,
    const float* Policy,
    float WinRate,
    float DrawRate,
    uint64_t Hash
) {
    bool NaNFound = false;
    if constexpr (NaNFallbackEnabled) {
        if (math::isnan_(WinRate)) {
            NaNFound = true;
            const Node* Parent = N->getParent();
            if (Parent == nullptr) {
                WinRate = 0.5f;
            } else {
                const double ParentWinRate =
                    Parent->getWinRateAccumulated() /
                    (Parent->getVisitsAndVirtualLoss() & Node::VisitMask);
                WinRate = (float)(1.0 - ParentWinRate);
            }
        }
        if (math::isnan_(DrawRate)) {
            NaNFound = true;
            const Node* Parent = N->getParent();
            if (Parent == nullptr) {
                DrawRate = 0.0f;
            } else {
                const double ParentDrawRate =
                    Parent->getDrawRateAccumulated() /
                    (Parent->getVisitsAndVirtualLoss() & Node::VisitMask);
                DrawRate = (float)ParentDrawRate;
            }
        }
    }

    assert(WinRate >= 0.0f && WinRate <= 1.0f);
    assert(DrawRate >= 0.0f && DrawRate <= 1.0f);

    const uint16_t NumChildren = N->getNumChildren();
    if (NumChildren == 1) {
        LegalPolicy[0] = 1.0f;
        N->setEvaluation(LegalPolicy, WinRate, DrawRate);
    } else {
        if constexpr (NaNFallbackEnabled) {
            for (uint16_t I = 0; I < NumChildren; ++I) {
                const std::size_t MoveIndex =
                    ml::getMoveIndex(SideToMove, N->getEdge()[I].getMove());
                LegalPolicy[I] = Policy[MoveIndex];
                if (math::isnan_(LegalPolicy[I])) {
                    NaNFound = true;
                    for (uint16_t J = 0; J < NumChildren; ++J) {
                        LegalPolicy[J] = 1.0f;
                    }
                    break;
                }
            }
        } else {
            for (uint16_t I = 0; I < NumChildren; ++I) {
                const std::size_t MoveIndex =
                    ml::getMoveIndex(SideToMove, N->getEdge()[I].getMove());
                LegalPolicy[I] = Policy[MoveIndex];
            }
        }
        ml::math::softmax_(LegalPolicy, NumChildren, 1.6f);
        N->setEvaluation(LegalPolicy, WinRate, DrawRate);
        N->sort();
    }

    N->updateAncestors(WinRate, DrawRate);

    if (ECache != nullptr && !NaNFound) {
        ECache->store(Hash, NumChildren, LegalPolicy, WinRate, DrawRate);
    }
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
