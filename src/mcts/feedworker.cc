//
// Copyright (c) 2025-2026 @nyashiki
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
            feedResult<true>(B->node(I), B->legalPolicy(I), B->winRate(I),
                             B->drawRate(I), B->hash(I));
        }
    } else {
        for (std::size_t I = 0; I < B->size(); ++I) {
            feedResult<false>(B->node(I), B->legalPolicy(I), B->winRate(I),
                              B->drawRate(I), B->hash(I));
        }
    }
}

template <bool NaNFallbackEnabled>
void FeedWorker::feedResult(Node* N, float* LegalPolicy, float WinRate,
                            float DrawRate, uint64_t Hash) {
    bool NaNFound = false;
    if constexpr (NaNFallbackEnabled) {
        const bool WinRateIsNaN = math::isnan_(WinRate);
        const bool DrawRateIsNaN = math::isnan_(DrawRate);
        if (WinRateIsNaN || DrawRateIsNaN) {
            NaNFound = true;
            double ParentExpectedScore = 0.5;
            double ParentDrawRate = 0.0;
            const Node* Parent = N->getParent();
            if (Parent != nullptr) {
                const uint64_t ParentVisits =
                    Parent->getVisitsAndVirtualLoss() & Node::VisitMask;
                if (ParentVisits > 0) {
                    ParentExpectedScore =
                        Parent->getExpectedScoreAccumulated() /
                        (double)ParentVisits;
                    ParentDrawRate = std::clamp(
                        Parent->getDrawRateAccumulated() / (double)ParentVisits,
                        0.0, 1.0);
                }
            }
            if (WinRateIsNaN) {
                // updateAncestors() expects a conditional win rate and will
                // convert it to a neutral score exactly once.
                WinRate =
                    (float)(1.0 - math::score::toConditionalWinRate(
                                      ParentExpectedScore, ParentDrawRate));
            }
            if (DrawRateIsNaN) {
                DrawRate = (float)ParentDrawRate;
            }
        }
    }

#ifndef NDEBUG
    if (WinRate < 0.0f || WinRate > 1.0f) {
        std::cerr << "Warning: WinRate is out of range: " << WinRate
                  << std::endl;
        std::abort();
    }
    if (DrawRate < 0.0f || DrawRate > 1.0f) {
        std::cerr << "Warning: DrawRate is out of range: " << DrawRate
                  << std::endl;
        std::abort();
    }
#endif

    const uint16_t NumChildren = N->getNumChildren();
    if (NumChildren == 1) {
        LegalPolicy[0] = 1.0f;
        N->setEvaluation(LegalPolicy, WinRate, DrawRate);
    } else {
        // LegalPolicy holds the logits of the legal moves in the edge
        // order, gathered by the evaluation worker; turn them into
        // probabilities in place.
        if constexpr (NaNFallbackEnabled) {
            for (uint16_t I = 0; I < NumChildren; ++I) {
                if (math::isnan_(LegalPolicy[I])) {
                    NaNFound = true;
                    for (uint16_t J = 0; J < NumChildren; ++J) {
                        LegalPolicy[J] = 1.0f;
                    }
                    break;
                }
            }
        }
        ml::math::softmax_(LegalPolicy, NumChildren, 1.0f);
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
