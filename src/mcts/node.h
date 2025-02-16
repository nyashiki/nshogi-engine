//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_MCTS_NODE_H
#define NSHOGI_ENGINE_MCTS_NODE_H

#include "../allocator/allocator.h"
#include "../math/fixedpoint.h"
#include "edge.h"
#include "pointer.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <nshogi/core/movelist.h>
#include <nshogi/core/state.h>
#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace mcts {

namespace {

void addAtomicDouble(std::atomic<double>* X, double T) {
    double Expected = X->load(std::memory_order_relaxed);
    double Desired;
    do {
        Desired = Expected + T;
    } while (!X->compare_exchange_weak(Expected, Desired));
}

} // namespace

struct Node {
 public:
    Node(Node* ParentNode)
        : Parent(ParentNode)
        , NumChildren(0)
        , VisitsAndVirtualLoss(0)
        , WinRateAccumulated(0)
        , DrawRateAccumulated(0)
        , PlyToTerminalSolved(0)
        , WinRatePredicted(0)
        , DrawRatePredicted(0)
        , SolverMove(0)
        , Repetition(core::RepetitionStatus::NoRepetition) {
    }

    static constexpr int VirtualLossShift = 48;
    static constexpr uint64_t VisitMask = (1ULL << VirtualLossShift) - 1;

    inline void resetParent(Node* P = nullptr) {
        Parent = P;
    }

    inline Node* getParent() {
        return Parent;
    }

    inline uint16_t getNumChildren() const {
        return NumChildren;
    }

    inline Pointer<Edge>& getEdge() {
        return Edges;
    }

    inline const Pointer<Edge>& getEdge() const {
        return Edges;
    }

    inline void incrementVirtualLoss() {
        constexpr uint64_t Value = 1ULL << VirtualLossShift;
        VisitsAndVirtualLoss.fetch_add(Value, std::memory_order_release);
    }

    inline void incrementVisits() {
        VisitsAndVirtualLoss.fetch_add(1, std::memory_order_release);
    }

    inline void incrementVisitsAndDecrementVirtualLoss() {
        constexpr uint64_t Value =
            (0xffffffffffffffffULL << VirtualLossShift) | 0b1ULL;
        VisitsAndVirtualLoss.fetch_add(Value, std::memory_order_release);
    }

    inline void decrementVirtualLoss() {
        constexpr uint64_t Value = 0xffffffffffffffffULL << VirtualLossShift;
        VisitsAndVirtualLoss.fetch_add(Value, std::memory_order_release);
    }

    inline uint64_t getVisitsAndVirtualLoss() const {
        return VisitsAndVirtualLoss.load(std::memory_order_acquire);
    }

    inline double getWinRateAccumulated() const {
        return WinRateAccumulated.load(std::memory_order_acquire);
    }

    inline double getDrawRateAccumulated() const {
        return DrawRateAccumulated.load(std::memory_order_acquire);
    }

    inline int16_t getPlyToTerminalSolved() const {
        return PlyToTerminalSolved.load(std::memory_order_acquire);
    }

    inline void setPlyToTerminalSolved(int16_t Ply) {
        PlyToTerminalSolved.store(Ply, std::memory_order_release);
    }

    inline float getWinRatePredicted() const {
        return WinRatePredicted;
    }

    inline float getDrawRatePredicted() const {
        return DrawRatePredicted;
    }

    inline int16_t expand(const nshogi::core::MoveList& MoveList,
                          allocator::Allocator* Allocator) {
        assert(Edges == nullptr);
        assert(MoveList.size() > 0);
        assert((VisitsAndVirtualLoss & VisitMask) == 0);

        Edges.mallocArray(Allocator, MoveList.size());
        if (Edges == nullptr) {
            // There is no available memory.
            return -1;
        }

        for (std::size_t I = 0; I < MoveList.size(); ++I) {
            Edges[I].setMove(core::Move16(MoveList[I]));
        }

        NumChildren = (uint16_t)MoveList.size();
        return (int16_t)NumChildren;
    }

    inline void setEvaluation(const float* Policy, float WinRate,
                              float DrawRate) {
        if (Policy != nullptr) {
            for (std::size_t I = 0; I < getNumChildren(); ++I) {
                Edges[I].setProbability(Policy[I]);
            }
        }

        WinRatePredicted = WinRate;
        DrawRatePredicted = DrawRate;
    }

    inline void sort() {
        std::sort(Edges.get(), Edges.get() + getNumChildren(),
                  [](const Edge& E1, const Edge& E2) {
                      return E1.getProbability() > E2.getProbability();
                  });
    }

    template <bool DecrementVirtualLoss = true>
    inline void updateAncestors(float WinRate, float DrawRate) {
        const float FlipWinRate = 1.0f - WinRate;
        bool Flip = false;

        Node* N = this;

        do {
            N->addWinRate(Flip ? FlipWinRate : WinRate);
            N->addDrawRate(DrawRate);

            if constexpr (DecrementVirtualLoss) {
                N->incrementVisitsAndDecrementVirtualLoss();
            } else {
                N->incrementVisits();
            }

            Flip = !Flip;
            N = N->getParent();
        } while (N != nullptr);
    }

    inline void addWinRate(double WinRate) {
        addAtomicDouble(&WinRateAccumulated, WinRate);
    }

    inline void addDrawRate(double DrawRate) {
        addAtomicDouble(&DrawRateAccumulated, DrawRate);
    }

    inline core::RepetitionStatus getRepetitionStatus() const {
        return Repetition;
    }

    inline void setRepetitionStatus(core::RepetitionStatus Value) {
        Repetition = Value;
    }

    Edge* mostPromisingEdgeV1() {
        const uint16_t NumChildren_ = getNumChildren();

        if (NumChildren_ == 0) {
            return nullptr;
        }

        const auto SMove = getSolverResult();

        uint32_t MostVisitedCount = 0;
        Edge* MostVisitedEdge = &getEdge()[0];

        for (uint16_t I = 0; I < NumChildren_; ++I) {
            Edge* E = &getEdge()[I];
            const Node* Child = E->getTarget();

            if (SMove == E->getMove()) {
                return E;
            }

            if (Child == nullptr) {
                continue;
            }

            if (Child->getPlyToTerminalSolved() < 0) {
                return E;
            }

            const uint32_t ChildVisits =
                (uint32_t)(Child->getVisitsAndVirtualLoss());

            if (Child->getPlyToTerminalSolved() > 0) {
                continue;
            }

            if (ChildVisits > MostVisitedCount) {
                MostVisitedCount = ChildVisits;
                MostVisitedEdge = E;
            }
        }

        return MostVisitedEdge;
    }

    Edge* mostPromisingEdgeV2() {
        const uint16_t NumChildren_ = getNumChildren();

        if (NumChildren_ == 0) {
            return nullptr;
        }

        const auto SMove = getSolverResult();

        double ScoreMax = 0;
        Edge* ScoreMaxEdge = &getEdge()[0];

        for (uint16_t I = 0; I < NumChildren_; ++I) {
            Edge* E = &getEdge()[I];
            const Node* Child = E->getTarget();

            if (SMove == E->getMove()) {
                return E;
            }

            if (Child == nullptr) {
                continue;
            }

            if (Child->getPlyToTerminalSolved() < 0) {
                return E;
            }

            const double ChildScore = Child->getWinRateAccumulated();

            if (ChildScore > ScoreMax) {
                ScoreMax = ChildScore;
                ScoreMaxEdge = E;
            }
        }

        return ScoreMaxEdge;
    }

    Edge* mostPromisingEdge() {
        return mostPromisingEdgeV1();
    }

    void setSolverResult(core::Move16 Move) {
        SolverMove.store(Move.value(), std::memory_order_release);
    }

    core::Move16 getSolverResult() const {
        return core::Move16::fromValue(
            SolverMove.load(std::memory_order_acquire));
    }

 private:
    Node* Parent;
    uint16_t NumChildren;
    Pointer<Edge> Edges;

    // Variables updated in search iteration.
    std::atomic<uint64_t> VisitsAndVirtualLoss;
    std::atomic<double> WinRateAccumulated;
    std::atomic<double> DrawRateAccumulated;
    std::atomic<int16_t> PlyToTerminalSolved;

    // Outputs of the evaluation function.
    float WinRatePredicted;
    float DrawRatePredicted;

    // Solver's result.
    std::atomic<uint16_t> SolverMove;

    core::RepetitionStatus Repetition;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_NODE_H
