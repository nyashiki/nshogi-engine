//
// Copyright (c) 2025-2026 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "searchworker.h"
#include "../globalconfig.h"

#include <cmath>
#include <limits>

#include <nshogi/core/movegenerator.h>

namespace nshogi {
namespace engine {
namespace mcts {

namespace {

void cancelVirtualLoss(Node* N) {
    do {
        N->decrementVirtualLoss();
        N = N->getParent();
    } while (N != nullptr);
}

} // namespace

SearchWorker::SearchWorker(allocator::Allocator* NodeAllocator,
                           allocator::Allocator* EdgeAllocator,
                           EvaluationQueue* EQ,
                           EvalCache* EC, Statistics* Stat)
    : worker::Worker(true)
    , NA(NodeAllocator)
    , EA(EdgeAllocator)
    , EQueue(EQ)
    , ECache(EC)
    , DfPnSolver(64)
    , PStat(Stat) {

    spawnThread();
}

SearchWorker::~SearchWorker() {
}

void SearchWorker::updateRoot(const core::State& S,
                              const core::StateConfig& StateConfig,
                              Node* Root) {
    State = std::make_unique<core::State>(S.clone());
    Config = StateConfig;
    RootNode = Root;
    RootSideToMove = State->getSideToMove();

    RootPly = State->getPly();
}

Node* SearchWorker::collectOneLeaf() {
    Node* CurrentNode = RootNode;

    while (true) {
        const uint64_t VisitsAndVirtualLossOld = CurrentNode->incrementVirtualLoss();
        const uint64_t Visits = VisitsAndVirtualLossOld & Node::VisitMask;

        if (Visits == 0) {
            if (VisitsAndVirtualLossOld == 0) {
                return CurrentNode;
            } else {
                // Another thread has collected this leaf node,
                // so there is nothing to do here.
                cancelVirtualLoss(CurrentNode);
                return nullptr;
            }
        }

        const uint16_t NumChildren = CurrentNode->getNumChildren();

        // If the number of visits of the node is larger than zero and
        // the number of children of it is zero, the state is a terminal state
        // and therefore the node is a leaf node.
        if (Visits > 0 && NumChildren == 0) {
            break;
        }

        // If the state is a repetition state, we regard the node as a leaf
        // node. But when the node is the root node, we don't so that the search
        // can proceed to have the children of the root in the tree.
        if (CurrentNode != RootNode &&
            CurrentNode->getRepetitionStatus() !=
                core::RepetitionStatus::NoRepetition) {
            break;
        }

        // If the ply of the state is equal to (or greater than) the max ply,
        // the game result is draw.
        if (State->getPly() >= Config.MaxPly) {
            break;
        }

        // If `getPlyToTerminalSolved()` returns a non-zero value,
        // it means the game theoretical value has already been solved and
        // therefore we don't have to search this node furthermore.
        if (CurrentNode->getPlyToTerminalSolved() != 0) {
            break;
        }

        Edge* E = computeUCBMaxEdge(CurrentNode, NumChildren, false);
        // computeUCBMaxEdge() can return nullptr if many threads reaches on the
        // same leaf node.
        if (E == nullptr) {
            PStat->incrementNumNullUCBMaxEdge();
            cancelVirtualLoss(CurrentNode);
            return nullptr;
        }

        State->doMove(State->getMove32FromMove16(E->getMove()));

        Node* Target = E->getTarget();
        // If `Target` is nullptr,
        // we have not extracted the child of this node.
        if (Target == nullptr) {
            const bool IsBeingExpanded = E->markExpanding();

            if (IsBeingExpanded) {
                PStat->incrementNumConflictNodeAllocation();
                cancelVirtualLoss(CurrentNode);
                return nullptr;
            }

            // Malloc a new node first before getting the lock for speed.
            Pointer<Node> NewNode;
            NewNode.malloc(NA, CurrentNode);

            if (NewNode == nullptr) {
                // If there is no available memory, it has failed to allocate a
                // new node.
                PStat->incrementNumFailedToAllocateNode();
                cancelVirtualLoss(CurrentNode);
                return nullptr;
            }

            auto* NewNodePtr = NewNode.get();
            NewNodePtr->incrementVirtualLoss();
            E->setTarget(std::move(NewNode));

            return NewNodePtr;
        }

        CurrentNode = E->getTarget();
    }

    return CurrentNode;
}

int16_t SearchWorker::expandLeaf(Node* LeafNode) {
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);
    const uint16_t NumMoves = (uint16_t)Moves.size();

    if (NumMoves == 0) {
        return 0;
    }

    return LeafNode->expand(Moves, EA);
}

void SearchWorker::immediateUpdateByWin(Node* LeafNode) {
    LeafNode->setEvaluation(nullptr, 1.0f, 0.0f);
    LeafNode->setPlyToTerminalSolved(1);
    LeafNode->updateAncestors(1.0f, 0.0f);
}

void SearchWorker::immediateUpdateByLoss(Node* LeafNode) {
    LeafNode->setEvaluation(nullptr, 0.0f, 0.0f);
    LeafNode->setPlyToTerminalSolved(-1);
    LeafNode->updateAncestors(0.0f, 0.0f);
}

void SearchWorker::immediateUpdateByDraw(Node* LeafNode, float DrawValue) {
    LeafNode->setEvaluation(nullptr, DrawValue, 1.0f);
    LeafNode->updateAncestors(DrawValue, 1.0f);
}

void SearchWorker::immediateUpdate(Node* LeafNode) {
    float WinRate = LeafNode->getWinRatePredicted();
    float DrawRate = LeafNode->getDrawRatePredicted();

    const auto RS = LeafNode->getRepetitionStatus();
    if (RS == core::RepetitionStatus::NoRepetition) {
        const auto PlyToTerminal = LeafNode->getPlyToTerminalSolved();
        if (PlyToTerminal > 0) {
            WinRate = 1.0;
            DrawRate = 0.0;
        } else if (PlyToTerminal < 0) {
            WinRate = 0.0;
            DrawRate = 0.0;
        }
    }

    LeafNode->updateAncestors(WinRate, DrawRate);
}

Edge* SearchWorker::computeUCBMaxEdge(Node* N, uint16_t NumChildren,
                                      bool regardNotVisitedWin) {
    assert(NumChildren > 0);
    const uint64_t CurrentVisitsAndVirtualLoss = N->getVisitsAndVirtualLoss();
    const uint64_t CurrentVisits =
        CurrentVisitsAndVirtualLoss & Node::VisitMask;
    uint64_t CurrentVirtualLoss =
        (CurrentVisitsAndVirtualLoss >> Node::VirtualLossShift) - 1; // Subtract 1 that is added in collectOneLeaf().

    if (CurrentVisits == 1) {
        // If the number of visit is equal to one, it means all children
        // is not expanded yet. Recall the UCB fomular, the most promising edge
        // is the edge with the highest prior. Since we have sorted the children
        // along their prior, we can simply select 0-th edge if the virtual loss
        // is zero.
        if (CurrentVirtualLoss < NumChildren) {
            if (CurrentVirtualLoss == 0) {
                return &N->getEdge()[0];
            } else {
                bool Acceptable = true;

                const double ThisPolicy =
                    (double)N->getEdge()[CurrentVirtualLoss].getProbability();
                const double Const =
                    1.0 / (CInit * std::sqrt((double)(CurrentVirtualLoss +
                                                      (uint64_t)1)));
                for (uint16_t I = 0; I < CurrentVirtualLoss - 1; ++I) {
                    const double Policy =
                        (double)N->getEdge()[I].getProbability();

                    if (Const + Policy / (double)CurrentVirtualLoss >=
                        ThisPolicy) {
                        Acceptable = false;
                        break;
                    }
                }

                if (Acceptable) {
                    return &N->getEdge()[CurrentVirtualLoss];
                } else {
                    PStat->incrementNumSpeculativeFailedEdge();
                    return nullptr;
                }
            }
        } else {
            // Otherwise, skip going deeper at this node.
            PStat->incrementNumTooManyVirtualLossEdge();
            return nullptr;
        }
    }

    const uint64_t CurrentVirtualVisits = CurrentVisits + CurrentVirtualLoss;

    const double Const =
        (std::log((double)(CurrentVirtualVisits + CBase) / (double)CBase) +
         CInit) *
        std::sqrt((double)CurrentVirtualVisits);

    Edge* UCBMaxEdge = nullptr;
    double UCBMaxValue = std::numeric_limits<double>::lowest();
    bool IsAllTargetLoss = true;
    int32_t WinTargetPlyMin = 10000;
    int32_t LossTargetPlyMax = 0;

    Edge* ShortestWinEdge = nullptr;
    Edge* LongestLossEdge = nullptr;

    assert(CurrentVisits > 1);
    assert(NumChildren > 0);
    for (uint16_t I = 0; I < NumChildren; ++I) {
        auto* const Edge = &N->getEdge()[I];
        auto* const Child = Edge->getTarget();

        const uint64_t ChildVisitsAndVirtualLoss =
            (Child != nullptr) ? Child->getVisitsAndVirtualLoss() : 0;

        // The child is not visited yet.
        if (Child == nullptr || ChildVisitsAndVirtualLoss == 0) {
            double UCBValue;
            const bool IsExpanding = Edge->isExpanding();
            if (IsExpanding) {
                // To consider the virtual loss, we simply
                // use 0.0 as the win rate of the child.
                UCBValue = Const * Edge->getProbability();
            } else {
                UCBValue = regardNotVisitedWin
                               ? (1.0 + Const * Edge->getProbability())
                               : (Const * Edge->getProbability());
            }

            // Since there is at least one unvisited child, which means
            // the child is not solved, we don't know all children are loss
            // states.
            IsAllTargetLoss = false;

            if (UCBValue > UCBMaxValue) {
                UCBMaxValue = UCBValue;
                UCBMaxEdge = Edge;
            }

            // Since the children is sorted by its policy,
            // we can break the loop here because if its visit count
            // is zero, a child that has higher policy has higher UCB value.
            // The relationship can be broken if the visit count is not zero,
            // but in that case, there is no unvisited child previously in this
            // loop.
            if (!IsExpanding) {
                break;
            } else {
                continue;
            }
        }

        const uint64_t ChildVisits =
            ChildVisitsAndVirtualLoss & Node::VisitMask;
        const uint64_t ChildVirtualLoss =
            ChildVisitsAndVirtualLoss >> Node::VirtualLossShift;

        if (ChildVisits == 0 && ChildVirtualLoss > 0) {
            // When the number of visits is zero and the number of
            // virtual loss is larger than zero of a node,
            // it means the node is being extracted.
            IsAllTargetLoss = false;
            PStat->incrementNumBeingExtractedChildren();
            continue;
        }

        const uint64_t ChildVirtualVisits = ChildVisits + ChildVirtualLoss;

        const int32_t PlyToTerminal =
            static_cast<int32_t>(Child->getPlyToTerminalSolved());
        if (PlyToTerminal > 0) {
            // If `PlyToTerminal` of a child is larger than zero,
            // it means the child is a win state from the perspective of
            // the side to move of the child state. Therefore, if we choose
            // this child, we will lose the game. When losing, one of the
            // best moves is a move that has longest sequence of moves to a
            // terminal.
            if (PlyToTerminal > LossTargetPlyMax) {
                LongestLossEdge = Edge;
                LossTargetPlyMax = PlyToTerminal;
            }
            continue;
        }

        IsAllTargetLoss = false;

        if (PlyToTerminal < 0) {
            const int32_t NegativePlyToTerminal = -PlyToTerminal;

            // If `PlyToTerminal` of a child is less than zero,
            // it means the child is a loss state from the perspective of
            // the side to move of the child state. Therefore, if we choose
            // this child, we will win the game. When winning, one of the
            // best moves is a move that has shortest sequence of moves to a
            // terminal.
            if (NegativePlyToTerminal < WinTargetPlyMin) {
                ShortestWinEdge = Edge;
                WinTargetPlyMin = NegativePlyToTerminal;
            }
            continue;
        }

        // Since we already have at least one winning edge,
        // if the current searching edge is not a winning edge, we don't have to
        // search it so `continue` here.
        if (ShortestWinEdge != nullptr) {
            continue;
        }

        assert(Child != nullptr);
        assert(ChildVirtualVisits > 0);
        const double ChildWinRate = computeWinRateOfChild(Child, ChildVisits, ChildVirtualVisits);
        const double UCBValue =
            ChildWinRate +
            Const * Edge->getProbability() / ((double)(1 + ChildVirtualVisits));

        if (UCBValue > UCBMaxValue) {
            UCBMaxValue = UCBValue;
            UCBMaxEdge = Edge;
        }
    }

    if (ShortestWinEdge != nullptr) {
        N->setPlyToTerminalSolved(static_cast<int16_t>(WinTargetPlyMin + 1));
        return ShortestWinEdge;
    }

    if (IsAllTargetLoss) {
        N->setPlyToTerminalSolved(static_cast<int16_t>(-LossTargetPlyMax - 1));
        return LongestLossEdge;
    }

    if (UCBMaxEdge == nullptr) {
        PStat->incrementNumUCBSelectionFailedEdge();
    }
    return UCBMaxEdge;
}

double SearchWorker::computeWinRateOfChild(
    Node* Child,
    uint64_t ChildVisits,
    uint64_t ChildVirtualVisits
) const {
    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAcuumulated = Child->getDrawRateAccumulated();

    const double WinRate =
        ((double)ChildVisits - ChildWinRateAccumulated) / (double)ChildVirtualVisits;
    const double DrawRate = ChildDrawRateAcuumulated / (double)ChildVisits;

    const double DrawValue = (State->getSideToMove() == core::Black)
                                 ? Config.BlackDrawValue
                                 : Config.WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

bool SearchWorker::doTask() {
    // Go back to the root state.
    while (State->getPly() != RootPly) {
        State->undoMove();
    }

    Node* LeafNode = collectOneLeaf();

    if (LeafNode == nullptr) {
        PStat->incrementNumNullLeaf();
        return false;
    }

    const uint64_t NumVisitsAndVirtualLoss =
        LeafNode->getVisitsAndVirtualLoss();
    const uint64_t NumVisits = NumVisitsAndVirtualLoss & Node::VisitMask;

    // The collected leaf node has been already evaluated.
    // This occurs when another thread has evaluated the leaf node,
    // the solver has solved the leaf node, or the leaf node is
    // game's terminal node (e.g., repetition).
    if (NumVisits > 0) {
        immediateUpdate(LeafNode);
        PStat->incrementNumNonLeaf();
        return false;
    }

    // Check repetition.
    if (LeafNode != RootNode) {
        const auto RS = State->getRepetitionStatus();
        LeafNode->setRepetitionStatus(RS);
        if (RS == core::RepetitionStatus::WinRepetition ||
            RS == core::RepetitionStatus::SuperiorRepetition) {
            immediateUpdateByWin(LeafNode);
            PStat->incrementNumRepetition();
            return false;
        } else if (RS == core::RepetitionStatus::LossRepetition ||
                   RS == core::RepetitionStatus::InferiorRepetition) {
            immediateUpdateByLoss(LeafNode);
            PStat->incrementNumRepetition();
            return false;
        } else if (RS == core::RepetitionStatus::Repetition) {
            immediateUpdateByDraw(LeafNode,
                                  State->getSideToMove() == core::Black
                                      ? Config.BlackDrawValue
                                      : Config.WhiteDrawValue);
            PStat->incrementNumRepetition();
            return false;
        }
    }

    const int16_t NumMoves = expandLeaf(LeafNode);
    // Check checkmate.
    if (NumMoves == 0) {
        PStat->incrementNumCheckmate();

        if (State->getPly(false) > 0) {
            const auto LastMove = State->getLastMove();
            if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                immediateUpdateByWin(LeafNode);
                return false;
            }
        }

        immediateUpdateByLoss(LeafNode);
        return false;
    }

    // This occurs when there is no available memory for edges.
    if (NumMoves == -1) {
        assert(LeafNode->getEdge() == nullptr);
        cancelVirtualLoss(LeafNode);
        PStat->incrementNumFailedToAllocateEdge();
        return false;
    }

    // Check delaration.
    if (State->canDeclare()) {
        immediateUpdateByWin(LeafNode);
        PStat->incrementNumCanDeclare();
        return false;
    }

    // Check the number of plies.
    if (State->getPly() >= Config.MaxPly) {
        immediateUpdateByDraw(LeafNode, State->getSideToMove() == core::Black
                                            ? Config.BlackDrawValue
                                            : Config.WhiteDrawValue);
        PStat->incrementNumOverMaxPly();
        return false;
    }

    // Check cache.
    bool CacheFound = false;
    if (ECache != nullptr) {
        CacheFound = ECache->load(*State, &CacheEvalInfo);

        if (CacheFound) {
            if (CacheEvalInfo.NumMoves == NumMoves) {
                LeafNode->setEvaluation(CacheEvalInfo.Policy,
                                        CacheEvalInfo.WinRate,
                                        CacheEvalInfo.DrawRate);
                LeafNode->sort();
                LeafNode->updateAncestors(CacheEvalInfo.WinRate,
                                          CacheEvalInfo.DrawRate);
                PStat->incrementNumCacheHit();
            } else {
                CacheFound = false;
            }
        }
    }

    // Evaluate the leaf node.
    if (!CacheFound) {
        const bool Succeeded = EQueue->add(*State, Config, LeafNode);

        if (Succeeded) {
            // Checkmate search.
            if (LeafNode->getSolverResult().isNone()) {
                const auto CheckmateSequence =
                    DfPnSolver.solveWithPV(
                        State.get(),
                        1000,
                        (uint64_t)std::min(64, Config.MaxPly - State->getPly())
                    );
                PStat->incrementNumSolverWorked();
                if (!CheckmateSequence.empty()) {
                    LeafNode->setSolverResult(core::Move16(CheckmateSequence[0]));
                    LeafNode->setPlyToTerminalSolved((int16_t)CheckmateSequence.size());
                } else {
                    LeafNode->setSolverResult(core::Move16::MoveInvalid());
                }
            }
        } else {
            // Our MCTS implementation marks a node as "in expansion" for speed:
            //     (VisitCount == 0 && VirtualLoss == 1)
            // Because the leaf was already expanded and a virtual loss was
            // propagated from the root to here, we have to roll both back.
            //
            // IMPORTANT: release the edges *before* cancelling the virtual
            // loss. If we cancelled the virtual loss first, the node would
            // momentarily have
            //     VisitCount == 0 && VirtualLoss == 0
            // while its edges were still present. Another thread could then
            // reach this leaf node and re-expand it, causing a data race.
            LeafNode->releaseEdges(EA);
            cancelVirtualLoss(LeafNode);
            PStat->incrementNumFailedToAddEvaluationQueue();
        }
    }

    return false;
}

SearchWorkerMaster::SearchWorkerMaster(
    const Context* C, allocator::Allocator* NodeAllocator,
    allocator::Allocator* EdgeAllocator, EvaluationQueue* EQueue,
    EvalCache* ECache, Statistics* Stat,
    std::function<void()> SearchStopCallback, std::shared_ptr<logger::Logger> L)
    : SearchWorker(NodeAllocator, EdgeAllocator, EQueue, ECache, Stat)
    , PContext(C)
    , Callback(SearchStopCallback)
    , Logger(std::move(L))
    , ImmediateLogEnabled(true)
    , Exiting(false) {

    StopCallThread = std::thread([this]() {
        while (true) {
            std::unique_lock<std::mutex> Lock(Mutex);

            StopCV.wait(Lock, [&]() { return Exiting || ToCallCallback; });

            if (Exiting) {
                break;
            }

            Callback();
            ToCallCallback = false;
        }
    });
}

SearchWorkerMaster::~SearchWorkerMaster() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Exiting = true;
    }

    StopCV.notify_one();
    StopCallThread.join();
}

void SearchWorkerMaster::setLimit(const engine::Limit& L) {
    Limit = L;
}

void SearchWorkerMaster::start() {
    SearchStartTime = std::chrono::steady_clock::now();
    MadeUpCheckElapsedPrevious = 0;
    NumNodesAtStart = RootNode->getVisitsAndVirtualLoss() & Node::VisitMask;
    LogOutputPrevious = 0;
    CallbackCalled.store(false, std::memory_order_release);
    ToCallCallback = false;

    SearchWorker::start();
}

bool SearchWorkerMaster::doTask() {
    // Dump log.
    const auto CurrentTime = std::chrono::steady_clock::now();
    const uint64_t Elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(CurrentTime -
                                                              SearchStartTime)
            .count());
    if (Elapsed >= PContext->getLogMargin() + LogOutputPrevious) {
        dumpPVLog(Elapsed);
        LogOutputPrevious = Elapsed;
    }

    if (CallbackCalled.load(std::memory_order_acquire)) {
        return false;
    }

    if (checkSearchToStop(Elapsed)) {
        issueStop();
        if (ImmediateLogEnabled || Elapsed > PContext->getLogMargin()) {
            dumpPVLog(Elapsed);
        }
        return false;
    }

    return SearchWorker::doTask();
}

void SearchWorkerMaster::issueStop() {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (isRunning()) {
        ToCallCallback = true;
        CallbackCalled.store(true, std::memory_order_release);
        StopCV.notify_one();
    }
}

void SearchWorkerMaster::enableImmediateLog() {
    ImmediateLogEnabled = true;
}

void SearchWorkerMaster::disableImmediateLog() {
    ImmediateLogEnabled = false;
}

logger::PVLog SearchWorkerMaster::getPVLog() const {
    logger::PVLog Log;

    Node* N = RootNode;

    const uint64_t Visits =
        RootNode->getVisitsAndVirtualLoss() & Node::VisitMask;

    Log.NumNodes = Visits;
    Log.CurrentSideToMove = RootSideToMove;
    Log.SolvedGameEndPly = N->getPlyToTerminalSolved();
    if (Log.SolvedGameEndPly > 0) {
        Log.WinRate = 1.0;
        Log.DrawRate = 0.0;
    } else if (Log.SolvedGameEndPly < 0) {
        Log.WinRate = 0.0;
        Log.DrawRate = 0.0;
    } else {
        if (Visits > 0) {
            Log.WinRate = N->getWinRateAccumulated() / (double)Visits;
            Log.DrawRate = N->getDrawRateAccumulated() / (double)Visits;
        } else {
            Log.WinRate = 0.0;
            Log.DrawRate = 0.0;
        }
    }
    Log.DrawValue = (State->getSideToMove() == core::Black)
                        ? Config.BlackDrawValue
                        : Config.WhiteDrawValue;

    while (N != nullptr) {
        Edge* E = N->mostPromisingEdge();

        if (E == nullptr) {
            break;
        }

        Log.PV.push_back(E->getMove());
        N = E->getTarget();
    }

    return Log;
}

void SearchWorkerMaster::dumpPVLog(uint64_t Elapsed) const {
    logger::PVLog Log = getPVLog();

    Log.ElapsedMilliSeconds = (uint32_t)Elapsed;
    if (Elapsed > 0) {
        Log.NodesPerSecond =
            (uint64_t)((double)(Log.NumNodes - NumNodesAtStart) * 1000ULL /
                       (double)Elapsed);
    }

    Logger->printPVLog(Log);
}

bool SearchWorkerMaster::isRootSolved() const {
    return RootNode->getPlyToTerminalSolved() != 0;
}

bool SearchWorkerMaster::checkNodeLimit() const {
    if (Limit.NumNodes == 0) {
        return false;
    }

    const uint64_t Visits =
        RootNode->getVisitsAndVirtualLoss() & Node::VisitMask;
    return Visits >= Limit.NumNodes;
}

bool SearchWorkerMaster::checkMemoryBudget() const {
    const double Factor = PContext->getMemoryLimitFactor();

    if (NA->getTotal() > 0 &&
        (double)NA->getUsed() >= (double)NA->getTotal() * Factor) {
        Logger->printLog("Memory limit (Node).");
        return true;
    }

    if (EA->getTotal() > 0 &&
        (double)EA->getUsed() >= (double)EA->getTotal() * Factor) {
        Logger->printLog("Memory limit (Edge).");
        return true;
    }

    return false;
}

bool SearchWorkerMaster::checkThinkingTimeBudget(uint64_t Elapsed) const {
    if (Elapsed >= PContext->getMaximumThinkingTimeMilliseconds()) {
        return true;
    }

    if (Elapsed < PContext->getMinimumThinkingTimeMilliseconds()) {
        return false;
    }

    if (Limit.isNoLimitAboutTime()) {
        return false;
    }

    const uint64_t Budget = Limit.TimeLimitMilliSeconds +
                            Limit.ByoyomiMilliSeconds +
                            Limit.IncreaseMilliSeconds;

    return Elapsed + PContext->getThinkingTimeMargin() >= Budget;
}

bool SearchWorkerMaster::hasMadeUpMind(uint64_t Elapsed) {
    if (Elapsed < MadeUpCheckElapsedPrevious + 470) {
        return false;
    }

    const uint16_t NumChildren = RootNode->getNumChildren();

    uint64_t SumVisits = 0;
    std::vector<double> Visits(NumChildren, 0.0);
    for (uint16_t I = 0; I < NumChildren; ++I) {
        Edge* E = &RootNode->getEdge()[I];
        Node* Child = E->getTarget();

        if (Child != nullptr) {
            const uint64_t V =
                Child->getVisitsAndVirtualLoss() & Node::VisitMask;
            SumVisits += V;
            Visits[I] = (double)V;
        }
    }

    if (SumVisits == 0) {
        return false;
    }

    for (uint16_t I = 0; I < NumChildren; ++I) {
        Visits[I] /= (double)SumVisits;
    }

    const Edge* BestEdge = RootNode->mostPromisingEdge();
    if (BestEdge == BestEdgePrevious &&
        Visits.size() == VisitsPrevious.size()) {
        double KLDivergence = 0.0;
        double KLDivergenceToPredicted = 0.0;

        for (std::size_t I = 0; I < Visits.size(); ++I) {
            if (VisitsPrevious[I] == 0.0) {
                continue;
            } else if (Visits[I] == 0.0) {
                KLDivergence = std::numeric_limits<double>::max();
                break;
            }

            const double Predicted = RootNode->getEdge()[I].getProbability();
            const double KLD =
                VisitsPrevious[I] * std::log(VisitsPrevious[I] / Visits[I]);

            KLDivergence += KLD;

            if (Predicted > 0) {
                const double KLDToP =
                    Predicted * std::log(Predicted / Visits[I]);
                KLDivergenceToPredicted += KLDToP;
            }
        }

        const double KLDThreshold =
            (KLDivergenceToPredicted < 0.4) ? 1e-5 : 1e-6;

        if (KLDivergence < KLDThreshold) {
            return true;
        }
    }

    // Save the variables for the next time.
    MadeUpCheckElapsedPrevious = Elapsed;
    BestEdgePrevious = BestEdge;
    VisitsPrevious = std::move(Visits);
    return false;
}

bool SearchWorkerMaster::checkSearchToStop(uint64_t Elapsed) {
    if (isRootSolved()) {
        return true;
    }

    if (checkNodeLimit()) {
        return true;
    }

    if (checkMemoryBudget()) {
        return true;
    }

    if (checkThinkingTimeBudget(Elapsed)) {
        Logger->printLog("Time limit.");
        return true;
    }

    if (!Limit.isNoLimitAboutTime()) {
        if (hasMadeUpMind(Elapsed)) {
            Logger->printLog("Made up mind.");
            return true;
        }
    }

    return false;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
