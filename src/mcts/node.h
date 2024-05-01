#ifndef NSHOGI_ENGINE_MCTS_NODE_H
#define NSHOGI_ENGINE_MCTS_NODE_H

#include "edge.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include "../allocator/allocator.h"
#include "../math/fixedpoint.h"
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

    inline Edge* getEdge(std::size_t I) {
        return &Edges[I];
    }

    inline void incrementVirtualLoss() {
        constexpr uint64_t Value = 1ULL << VirtualLossShift;
        VisitsAndVirtualLoss.fetch_add(Value, std::memory_order_relaxed);
    }

    inline void incrementVisitsAndDecrementVirtualLoss() {
        constexpr uint64_t Value = (0xffffffffffffffffULL << VirtualLossShift) | 0b1ULL;
        VisitsAndVirtualLoss.fetch_add(Value, std::memory_order_relaxed);
    }

    inline uint64_t getVisitsAndVirtualLoss() {
        return VisitsAndVirtualLoss.load(std::memory_order_relaxed);
    }

    inline double getWinRateAccumulated() const {
        return WinRateAccumulated.load(std::memory_order_relaxed);
    }

    inline double getDrawRateAccumulated() const {
        return DrawRateAccumulated.load(std::memory_order_relaxed);
    }

    inline int16_t getPlyToTerminalSolved() const {
        return PlyToTerminalSolved.load(std::memory_order_relaxed);
    }

    inline void setPlyToTerminalSolved(int16_t Ply) {
        PlyToTerminalSolved.store(Ply, std::memory_order_relaxed);
    }

    inline float getWinRatePredicted() const {
        return WinRatePredicted;
    }

    inline float getDrawRatePredicted() const {
        return DrawRatePredicted;
    }

    inline void expand(const nshogi::core::MoveList& MoveList) {
        assert(Edges == nullptr);

        Edges = std::make_unique<Edge[]>(MoveList.size());
        assert(Edges != nullptr);

        for (std::size_t I = 0; I < MoveList.size(); ++I) {
            Edges[I].setMove(core::Move16(MoveList[I]));
        }

        NumChildren = (uint16_t)MoveList.size();
    }

    inline void setEvaluation(const float* Policy, float WinRate, float DrawRate) {
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
        Edge* MostVisitedEdge = getEdge(0);

        for (uint16_t I = 0; I < NumChildren_; ++I) {
            Edge* E = getEdge(I);
            Node* Child = E->getTarget();

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
        Edge* ScoreMaxEdge = getEdge(0);

        for (uint16_t I = 0; I < NumChildren_; ++I) {
            Edge* E = getEdge(I);
            Node* Child = E->getTarget();

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

    void setSolverResult(const core::Move16& Move) {
        SolverMove.store(Move.value(), std::memory_order_relaxed);
    }

    core::Move16 getSolverResult() const {
        return core::Move16::fromValue(SolverMove.load(std::memory_order_relaxed));
    }

    void* operator new(std::size_t Size) {
        return allocator::getNodeAllocator().malloc(Size);
    }

    void operator delete(void* Ptr) noexcept {
        return allocator::getNodeAllocator().free(Ptr);
    }

    void* operator new[](std::size_t) = delete;
    void operator delete[](void*) = delete;

 private:
    Node* Parent;
    uint16_t NumChildren;
    std::unique_ptr<Edge[]> Edges;

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
