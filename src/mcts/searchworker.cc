#include "searchworker.h"
#include "mutexpool.h"
#include "../globalconfig.h"

#include <cassert>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <nshogi/solver/mate1ply.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/ml/math.h>

namespace nshogi {
namespace engine {
namespace mcts {

SearchWorker::SearchWorker(std::size_t BatchSize, evaluate::Evaluator* Ev, CheckmateSearcher* CSearcher_, MutexPool<lock::SpinLock>* MPool, EvalCache* ECache_)
    : BatchSizeMax(BatchSize), Batch(BatchSize, Ev), IsRunning(false), IsFinnishing(false), CSearcher(CSearcher_), MtxPool(MPool), ECache(ECache_) {

    WorkerThread = std::thread(&SearchWorker::mainLoop, this);
}

SearchWorker::~SearchWorker() {
}

void SearchWorker::start(Node* Root, const core::State& St, const core::StateConfig& Config) {
    LeafInfos.clear();
    for (std::size_t I = 0; I < BatchSizeMax; ++I) {
        LeafInfos.push_back({St.clone(), nullptr});
    }

    RootNode = Root;
    RootPly = St.getPly();
    StateConfig = &Config;

    std::lock_guard<std::mutex> Lock(Mtx);
    assert(!IsRunning);
    IsRunning = true;
    IsRunningInternal_ = true;

    Cv.notify_one();
}

void SearchWorker::stop() {
    {
        std::lock_guard<std::mutex> Lock(Mtx);

        assert(IsRunning);
        IsRunning.store(false);
    }

    Cv.notify_one();
}

void SearchWorker::await() {
    std::unique_lock<std::mutex> Lock(Mtx);

    AwaitCv.wait(Lock, [&]{
        return !IsRunningInternal_;
    });
}

void SearchWorker::join() {
    {
        std::lock_guard<std::mutex> Lock(Mtx);
        IsFinnishing = true;
        Cv.notify_one();
    }

    WorkerThread.join();
}

void SearchWorker::mainLoop() {
    while (true) {
        {
            std::unique_lock<std::mutex> Lock(Mtx);

            if (!IsRunning.load(std::memory_order_relaxed)) {
                IsRunningInternal_ = false;
                AwaitCv.notify_one();
            }

            Cv.wait(Lock, [&]{
                return IsRunningInternal_
                       || IsRunning.load(std::memory_order_relaxed)
                       || IsFinnishing;
            });

            if (IsFinnishing) {
                break;
            }

            IsRunningInternal_ = true;
        }

        while (IsRunning.load(std::memory_order_relaxed)) {
            doOneIteration();
        }
    }
}

void SearchWorker::doOneIteration() {
    Batch.reset();

    uint16_t OutOfOrder = 0;
    uint16_t SequentialNullTerminal = 0;

    std::size_t BatchIndex = 0;

    while (true) {
        if (Batch.size() >= Batch.getBatchSizeMax()) {
            break;
        }

        if (OutOfOrder >= Batch.getBatchSizeMax()) {
            break;
        }

        if (SequentialNullTerminal >= 3 && BatchIndex > 0) {
            break;
        }

        // Restore the state to the root node.
        goBack(&LeafInfos[BatchIndex].State);

        // Select a leaf node.
        Node* LeafNode = selectLeafNode(&LeafInfos[BatchIndex].State);

        if (LeafNode == nullptr) {
            ++OutOfOrder;
            if (BatchIndex > 0) {
                ++SequentialNullTerminal;
            }
            continue;
        }

        SequentialNullTerminal = 0;

        const uint64_t LeafVisits = LeafNode->getVisitsAndVirtualLoss() & Node::VisitMask;

        if (LeafVisits > 0) {
            // A terminal node has been selected.
            updateAncestors(LeafNode);
            ++OutOfOrder;
            continue;
        }

        if (evaluateByRule(LeafInfos[BatchIndex].State, LeafNode)) {
            ++OutOfOrder;
            continue;
        }

        LeafInfos[BatchIndex].LeafNode = LeafNode;

        if (ECache != nullptr) {
            bool CacheExists = ECache->load(LeafInfos[BatchIndex].State, &EInfo);

            if (CacheExists) {
                const auto MoveList = core::MoveGenerator::generateLegalMoves(LeafInfos[BatchIndex].State);

                assert((LeafNode->getVisitsAndVirtualLoss() & Node::VisitMask) == 0);
                assert(MoveList.size() > 0);

                LeafNode->expand(MoveList);
                feedLeafNode<false>(LeafInfos[BatchIndex], EInfo.Policy, EInfo.WinRate, EInfo.DrawRate);
                updateAncestors(LeafNode);

                ++OutOfOrder;
                continue;
            }
        }

        // Feed batch with the leaf node.
        assert(BatchIndex < Batch.getBatchSizeMax());
        Batch.add(LeafInfos[BatchIndex].State, *StateConfig);

        if (CSearcher != nullptr) {
            if (LeafInfos[BatchIndex].LeafNode->getSolverResult().isNone()) {
                CSearcher->addTask(LeafInfos[BatchIndex].LeafNode,
                        LeafInfos[BatchIndex].State.getPosition());
            }
        }

        ++BatchIndex;
    }

    if (BatchIndex > 0) {
        evaluateStoredLeafNodes(BatchIndex);
    }
}

void SearchWorker::evaluateStoredLeafNodes(std::size_t BatchIndexMax) {
    Batch.doInferenceNonBlocking();

    // Here, we have free time until the computation is done.
    for (std::size_t I = 0; I < BatchIndexMax; ++I) {
        const auto MoveList = core::MoveGenerator::generateLegalMoves(LeafInfos[I].State);

        // No legal moves, which means the king is checkmated.
        if (MoveList.size() == 0) {
            // If the last opponnet move is dropping a pawn, it's illegal.
            if (LeafInfos[I].State.getPly() > 0) {
                const auto& LastMove = LeafInfos[I].State.getLastMove();
                if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                    immediateUpdateByWin(LeafInfos[I].LeafNode);
                    LeafInfos[I].LeafNode = nullptr;
                    continue;
                }
            }

            immediateUpdateByLoss(LeafInfos[I].LeafNode);
            LeafInfos[I].LeafNode = nullptr;
            continue;
        }

        // Extract children of the leaf node.
        assert(LeafInfos[I].LeafNode->getVisitsAndVirtualLoss() == (1ULL << Node::VirtualLossShift));
        LeafInfos[I].LeafNode->expand(MoveList);
    }

    Batch.await();

    for (std::size_t I = 0; I < Batch.size(); ++I) {
        if (LeafInfos[I].LeafNode == nullptr) {
            continue;
        }

        feedLeafNode<true>(LeafInfos[I], Batch.getPolicy(I), Batch.getWinRate(I), Batch.getDrawRate(I));
        updateAncestors(LeafInfos[I].LeafNode);
    }
}

bool SearchWorker::evaluateByRule(const core::State& State, Node* N) {
    if ((StateConfig->Rule & core::Declare27_ER) != 0) {
        if (State.canDeclare()) {
            immediateUpdateByWin(N);
            return true;
        }
    }

    if (State.getPly() >= StateConfig->MaxPly) {
        immediateUpdateByDraw(N);
        return true;
    }

    // Check win/draw/loss by the rule.
    if (N != RootNode) {
        core::RepetitionStatus Repetition = State.getRepetitionStatus();

        N->setRepetitionStatus(Repetition);
        if (Repetition == core::RepetitionStatus::WinRepetition ||
                Repetition == core::RepetitionStatus::SuperiorRepetition) {

            immediateUpdateByWin(N);
            return true;
        } else if (Repetition == core::RepetitionStatus::LossRepetition ||
                Repetition == core::RepetitionStatus::InferiorRepetition) {

            immediateUpdateByLoss(N);
            return true;
        } else if (Repetition == core::RepetitionStatus::Repetition) {
            immediateUpdateByDraw(N);
            return true;
        }
    }

    return false;
}

template <bool PolicyLogits>
void SearchWorker::feedLeafNode(const LeafInfo& LInfo, const float* Policy, float WinRate, float DrawRate) {
    if constexpr (PolicyLogits) {
        float LegalPolicy[ml::MoveIndexMax];
        for (uint16_t I = 0; I < LInfo.LeafNode->getNumChildren(); ++I) {
            const std::size_t MoveIndex = ml::getMoveIndex(LInfo.State.getSideToMove(), LInfo.LeafNode->getEdge(I)->getMove());
            LegalPolicy[I] = Policy[MoveIndex];
        }

        // ml::math::softmax_(LegalPolicy, LInfo.LeafNode->getNumChildren(), 1.6f);
        ml::math::softmax_(LegalPolicy, LInfo.LeafNode->getNumChildren(), 1.2062533236123854f);

        LInfo.LeafNode->setEvaluation(LegalPolicy, WinRate, DrawRate);

        // Store the evaluation result into the cache.
        if (ECache != nullptr && LInfo.LeafNode->getNumChildren() < EvalCache::MAX_CACHE_MOVES_COUNT) {
            assert(LInfo.LeafNode->getNumChildren() > 0);
            ECache->store(LInfo.State, LInfo.LeafNode->getNumChildren(),
                    LegalPolicy, WinRate, DrawRate);
        }
    } else {
        LInfo.LeafNode->setEvaluation(Policy, WinRate, DrawRate);
    }

    LInfo.LeafNode->sort();
}

void SearchWorker::goBack(core::State* S) const {
    while (S->getPly() != RootPly) {
        S->undoMove();
    }
}

void SearchWorker::updateAncestors(Node* N) const {
    float LeafWinRate = N->getWinRatePredicted();
    float LeafDrawRate = N->getDrawRatePredicted();

    if (N->getPlyToTerminalSolved() > 0) {
        LeafWinRate = 1.0f;
        LeafDrawRate = 0.0f;
    } else if (N->getPlyToTerminalSolved() < 0) {
        LeafWinRate = 0.0f;
        LeafDrawRate = 0.0f;
    } else if (N->getRepetitionStatus() == core::RepetitionStatus::Repetition) {
        LeafWinRate = 0.5f;
        LeafDrawRate = 1.0f;
    }

    const float FlipWinRate = 1.0f - LeafWinRate;
    bool Flip = false;

    while (N != nullptr) {
        N->addWinRate(Flip? FlipWinRate : LeafWinRate);
        N->addDrawRate(LeafDrawRate);

        N->incrementVisitsAndDecrementVirtualLoss();

        Flip = !Flip;
        N = N->getParent();
    }
}

void SearchWorker::immediateUpdateByWin(Node* N) const {
    N->setEvaluation(nullptr, 1.0f, 0.0f);
    N->setPlyToTerminalSolved(1);
    updateAncestors(N);
}

void SearchWorker::immediateUpdateByLoss(Node* N) const {
    N->setEvaluation(nullptr, 0.0f, 0.0f);
    N->setPlyToTerminalSolved(-1);
    updateAncestors(N);
}

void SearchWorker::immediateUpdateByDraw(Node* N) const {
    N->setEvaluation(nullptr, 0.0f, 1.0f);
    updateAncestors(N);
}

Node* SearchWorker::selectLeafNode(core::State* S) const {
    Node* N = RootNode;
    uint16_t Depth = 0;

    while (true) {
        const uint64_t Visits = N->getVisitsAndVirtualLoss() & Node::VisitMask;

        if (CSearcher != nullptr) {
            if (N->getSolverResult().isNone()) {
                CSearcher->addTask(N, S->getPosition());
            }
        }

        if (Visits == 0) {
            if (N == RootNode) {
                if (MtxPool != nullptr) {
                    MtxPool->getRootMtx()->lock();
                }

                if (N->getVisitsAndVirtualLoss() != 0ULL) {
                    if (MtxPool != nullptr) {
                        MtxPool->getRootMtx()->unlock();
                    }

                    return nullptr;
                }

                N->incrementVirtualLoss();

                if (MtxPool != nullptr) {
                    MtxPool->getRootMtx()->unlock();
                }

                return N;
            }

            return nullptr;
        }

        if (Visits > 0 && N->getNumChildren() == 0) {
            break;
        }

        if (N != RootNode && N->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
            break;
        }

        if (N->getPlyToTerminalSolved() != 0) {
            break;
        }

        Edge* E = computeUCBMaxEdge(*S, N, Depth <= 1);
        if (E == nullptr) {
            return nullptr;
        }

        S->doMove(S->getMove32FromMove16(E->getMove()));

        Node* Target = E->getTarget();

        if (Target == nullptr) {
            lock::SpinLock* EdgeMtx = nullptr;
            if (MtxPool != nullptr) {
                EdgeMtx = MtxPool->get(reinterpret_cast<void*>(E));
                EdgeMtx->lock();

                if (E->getTarget() != nullptr) {
                    // This thread has reached a leaf node but another thread also
                    // had reached this leaf node and has evaluated this leaf node.
                    // Therefore E->getTarget() is not nullptr anymore and
                    // nothing to do is left.
                    EdgeMtx->unlock();
                    return nullptr;
                }
            }

            std::unique_ptr<Node> NewNode = std::make_unique<Node>(N);
            Node* NewNodePtr = NewNode.get();

            if (NewNodePtr == nullptr) {
                // This happens when memory allocation is failed.

                if (MtxPool != nullptr) {
                    EdgeMtx->unlock();
                }

                return nullptr;
            }

            incrementVirtualLoss(NewNodePtr);

            E->setTarget(std::move(NewNode));

            if (EdgeMtx != nullptr) {
                EdgeMtx->unlock();
            }

            return NewNodePtr;
        }

        N = E->getTarget();
        ++Depth;
    }

    incrementVirtualLoss(N);
    return N;
}

Edge* SearchWorker::computeUCBMaxEdge(const core::State& State, Node* N, bool RegardNotVisitedWin) const {
    const uint16_t NumChildren = N->getNumChildren();
    assert(NumChildren > 0);

    const uint64_t CurrentVisitsAndVirtualLoss = N->getVisitsAndVirtualLoss();
    const uint64_t CurrentVisit = CurrentVisitsAndVirtualLoss & Node::VisitMask;
    const uint32_t CurrentVirtualLoss = (uint32_t)(CurrentVisitsAndVirtualLoss >> Node::VirtualLossShift);

    if (CurrentVisit == 1) {
        if (CurrentVirtualLoss < NumChildren) {
            // Since the edges are sorted by its policy,
            // the most promising edge when CurrentVisit == 1 is Edges[0];
            return N->getEdge(CurrentVirtualLoss);
        } else {
            return nullptr;
        }
    }

    const uint64_t CurrentVirtualVisits = CurrentVisit + CurrentVirtualLoss;

    const double Const = std::log((double)(CurrentVirtualVisits + CBase) / (double)CBase + CInit) *
                        std::sqrt((double)CurrentVirtualVisits);

    Edge* UCBMaxEdge = nullptr;
    double UCBMaxValue = -1e9;
    bool IsAllTargetLoss = true;
    int16_t WinTargetPlyMin = 1024;
    int16_t LossTargetPlyMax = 0;

    Edge* ShortestWinEdge = nullptr;
    Edge* LongestLossEdge = nullptr;

    for (uint16_t I = 0; I < NumChildren; ++I) {
        auto* Edge = N->getEdge(I);
        auto* Child = Edge->getTarget();

        double UCBValue;

        if (Child == nullptr) {
            UCBValue =
                RegardNotVisitedWin ? (1.0 + Const * Edge->getProbability())
                                    : Const * Edge->getProbability();
            IsAllTargetLoss = false;
        } else {
            const uint64_t ChildVisitsAndVirtualLoss = Child->getVisitsAndVirtualLoss();
            const uint64_t ChildVisits = ChildVisitsAndVirtualLoss & Node::VisitMask;
            const uint32_t ChildVirtualLoss = (uint32_t)(ChildVisitsAndVirtualLoss >> Node::VirtualLossShift);
            const uint64_t ChildVirtualVisits = ChildVisits + ChildVirtualLoss;

            if (ChildVisits == 0 && ChildVirtualLoss > 0) {
                // This node is being evaluated so don't touch it now.
                IsAllTargetLoss = false;
                continue;
            }

            const int16_t PlyToTerminal = Child->getPlyToTerminalSolved();

            if (PlyToTerminal > 0) {
                if (PlyToTerminal > LossTargetPlyMax) {
                    LongestLossEdge = Edge;
                    LossTargetPlyMax = PlyToTerminal;
                }

                continue;
            }

            IsAllTargetLoss = false;

            if (PlyToTerminal < 0) {
                const int16_t NegativePlyToTerminal = -PlyToTerminal;

                if (NegativePlyToTerminal < WinTargetPlyMin) {
                    ShortestWinEdge = Edge;
                    WinTargetPlyMin = NegativePlyToTerminal;
                }

                continue;
            }

            if (ShortestWinEdge != nullptr) {
                continue;
            }

            const double ChildWinRate = computeWinRateOfChild(State, Child, ChildVisits, ChildVirtualVisits);
            UCBValue = ChildWinRate + Const * Edge->getProbability() / ((double)(1 + ChildVirtualVisits));
        }

        if (UCBValue > UCBMaxValue) {
            UCBMaxValue = UCBValue;
            UCBMaxEdge = Edge;
        }
    }

    if (ShortestWinEdge != nullptr) {
        N->setPlyToTerminalSolved(WinTargetPlyMin + 1);
        return ShortestWinEdge;
    }

    if (IsAllTargetLoss) {
        N->setPlyToTerminalSolved(-LossTargetPlyMax - 1);
        return LongestLossEdge;
    }

    return UCBMaxEdge;
}

void SearchWorker::incrementVirtualLoss(Node* N) const {
    while (N != nullptr) {
        N->incrementVirtualLoss();
        N = N->getParent();
    }
}

double SearchWorker::computeWinRateOfChild(const core::State& State, Node* Child, uint64_t ChildVisits, uint64_t ChildVirtualVisits) const {
    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAccumulated = Child->getDrawRateAccumulated();

    const double WinRate = ((double)ChildVisits - ChildWinRateAccumulated) / (double)ChildVirtualVisits;
    const double DrawRate = ChildDrawRateAccumulated / (double)ChildVirtualVisits;

    const auto SideToMove = State.getSideToMove();
    const double DrawValue = (SideToMove == core::Black)? StateConfig->BlackDrawValue : StateConfig->WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
