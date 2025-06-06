//
// Copyright (c) 2025 @nyashiki
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
                                     CheckmateQueue* CQ,
                                     MutexPool<>* MP,
                                     EvalCache* EC,
                                     Statistics* Stat)
    : worker::Worker(true)
    , NA(NodeAllocator)
    , EA(EdgeAllocator)
    , EQueue(EQ)
    , CQueue(CQ)
    , MtxPool(MP)
    , ECache(EC)
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

    RootPly = State->getPly();
}

Node* SearchWorker::collectOneLeaf() {
    using MtxLockType = typename std::remove_reference_t<decltype(*MtxPool)>::LockType;

    Node* CurrentNode = RootNode;

    while (true) {
        const uint64_t VisitsAndVirtualLoss =
            CurrentNode->getVisitsAndVirtualLoss();
        const uint64_t Visits = VisitsAndVirtualLoss & Node::VisitMask;

        if (CQueue != nullptr) {
            // If checkmate searcher is enabled and the node has not been
            // tried to solve, feed the node into the checkmate searcher.
            if (CurrentNode->getSolverResult().isNone()) {
                CQueue->tryAdd(CurrentNode, State->getPosition());
            }
        }

        if (Visits == 0) {
            const uint64_t VirtualLoss =
                VisitsAndVirtualLoss >> Node::VirtualLossShift;

            if (VirtualLoss == 0) {
                std::unique_lock<MtxLockType> Lock;
                if (MtxPool != nullptr) {
                    const uint64_t NodeMtxHash = reinterpret_cast<uint64_t>(CurrentNode);
                    Lock = std::unique_lock<MtxLockType>(*MtxPool->get(NodeMtxHash));
                }

                // Re-check the number of visit and virtual loss after getting a
                // lock. It is no longer zero if another thread has expanded
                // this leaf.
                if (CurrentNode->getVisitsAndVirtualLoss() != 0ULL) {
                    return nullptr;
                }

                incrementVirtualLosses(CurrentNode);
                return CurrentNode;
            }

            return nullptr;
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
            return nullptr;
        }

        State->doMove(State->getMove32FromMove16(E->getMove()));

        Node* Target = E->getTarget();
        if (Target == nullptr) {
            // If `Target` is nullptr, we have not extracted the child of this
            // node.

            // Malloc a new node first before getting the lock for speed.
            Pointer<Node> NewNode;
            NewNode.malloc(NA, CurrentNode);

            if (NewNode == nullptr) {
                // If there is no available memory, it has failed to allocate a
                // new node.
                PStat->incrementNumFailedToAllocateNode();
                return nullptr;
            }

            std::unique_lock<MtxLockType> Lock;
            if (MtxPool != nullptr) {
                const uint64_t EdgeMtxHash = reinterpret_cast<uint64_t>(E);
                Lock = std::unique_lock<MtxLockType>(*MtxPool->get(EdgeMtxHash));

                if (E->getTarget() != nullptr) {
                    // This thread has reached a leaf node but another thread
                    // also had reached this leaf node and has evaluated this
                    // leaf node. Therefore E->getTarget() is no longer nullptr
                    // and nothing to do is left.
                    NewNode.destroy(NA, 1);
                    PStat->incrementNumConflictNodeAllocation();
                    return nullptr;
                }
            }

            auto* NewNodePtr = NewNode.get();
            incrementVirtualLosses(NewNodePtr);
            E->setTarget(std::move(NewNode));
            return NewNodePtr;
        }

        CurrentNode = E->getTarget();
    }

    incrementVirtualLosses(CurrentNode);
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

void SearchWorker::immediateUpdateByDraw(Node* LeafNode,
                                                  float DrawValue) {
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
    const uint64_t CurrentVirtualLoss =
        CurrentVisitsAndVirtualLoss >> Node::VirtualLossShift;

    if (CurrentVisits == 1) {
        // If the number of visit is equal to one, it means all children
        // is not expanded yet. Recall the UCB fomular, the most promising edge
        // is the edge with the highest prior. Since we have sorted the children
        // along their prior, we can simply select 0-th edge if the virtual loss
        // is zero.
        if (CurrentVirtualLoss < NumChildren) {
            if (CurrentVirtualLoss == 0) {
                PStat->incrementNumPolicyGreedyEdge();
                return &N->getEdge()[0];
            } else {
                bool Acceptable = true;

                const double ThisPolicy = (double)N->getEdge()[CurrentVirtualLoss].getProbability();
                const double Const = 1.0 / (CInit * std::sqrt((double)CurrentVirtualLoss + 1.0));
                for (uint16_t I = 0; I < CurrentVirtualLoss - 1; ++I) {
                    const double Policy = (double)N->getEdge()[I].getProbability();

                    if (Const + Policy / (double)CurrentVirtualLoss >= ThisPolicy) {
                        Acceptable = false;
                        break;
                    }
                }

                if (Acceptable) {
                    PStat->incrementNumSpeculativeEdge();
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
        std::log((double)(CurrentVirtualVisits + CBase) / (double)CBase +
                 CInit) *
        std::sqrt((double)CurrentVirtualVisits);

    Edge* UCBMaxEdge = nullptr;
    double UCBMaxValue = std::numeric_limits<double>::lowest();
    bool IsAllTargetLoss = true;
    int16_t WinTargetPlyMin = 1024;
    int16_t LossTargetPlyMax = 0;

    Edge* ShortestWinEdge = nullptr;
    Edge* LongestLossEdge = nullptr;

    for (uint16_t I = 0; I < NumChildren; ++I) {
        auto* const Edge = &N->getEdge()[I];
        auto* const Child = Edge->getTarget();

        // The child is not visited yet.
        if (Child == nullptr) {
            const double UCBValue = regardNotVisitedWin
                                        ? (1.0 + Const * Edge->getProbability())
                                        : (Const * Edge->getProbability());

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
            PStat->incrementNumFirstUnvisitedChildEdge();
            break;
        }

        const uint64_t ChildVisitsAndVirtualLoss =
            Child->getVisitsAndVirtualLoss();
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

        const int16_t PlyToTerminal = Child->getPlyToTerminalSolved();
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
            const int16_t NegativePlyToTerminal = -PlyToTerminal;

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

        const double ChildWinRate =
            computeWinRateOfChild(Child, ChildVisits, ChildVirtualVisits);
        const double UCBValue =
            ChildWinRate +
            Const * Edge->getProbability() / ((double)(1 + ChildVirtualVisits));

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

    if (UCBMaxEdge == nullptr) {
        PStat->incrementNumUCBSelectionFailedEdge();
    }
    return UCBMaxEdge;
}

double SearchWorker::computeWinRateOfChild(Node* Child, uint64_t ChildVisits,
                                              uint64_t ChildVirtualVisits) {
    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAcuumulated = Child->getDrawRateAccumulated();

    const double WinRate = ((double)ChildVisits - ChildWinRateAccumulated) /
                           (double)ChildVirtualVisits;
    const double DrawRate =
        ChildDrawRateAcuumulated / (double)ChildVirtualVisits;

    const double DrawValue = (State->getSideToMove() == core::Black)
                                 ? Config.BlackDrawValue
                                 : Config.WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

void SearchWorker::incrementVirtualLosses(Node* N) {
    do {
        N->incrementVirtualLoss();
        N = N->getParent();
    } while (N != nullptr);
}

bool SearchWorker::doTask() {
    // Go back to the root state.
    while (State->getPly() != RootPly) {
        State->undoMove();
    }

    Node* LeafNode = collectOneLeaf();

    if (LeafNode == nullptr) {
        // std::this_thread::yield();
        PStat->incrementNumNullLeaf();
        return isRunning();
    }

    const uint64_t NumVisitsAndVirtualLoss =
        LeafNode->getVisitsAndVirtualLoss();
    const uint64_t NumVisits = NumVisitsAndVirtualLoss & Node::VisitMask;
    const uint64_t VirtualLoss =
        NumVisitsAndVirtualLoss >> Node::VirtualLossShift;

    // Collected leafnode has already evaluated.
    // This occurs when another thread has evaluated the leaf node,
    // the solver has solved the leaf node, or the leaf node is
    // game's terminal node e.g., repetition.
    if (NumVisits > 0) {
        // std::this_thread::yield();
        immediateUpdate(LeafNode);
        PStat->incrementNumNonLeaf();
        return isRunning();
    }

    assert(VirtualLoss == 1);

    // Check repetition.
    if (LeafNode != RootNode) {
        const auto RS = State->getRepetitionStatus();
        LeafNode->setRepetitionStatus(RS);
        if (RS == core::RepetitionStatus::WinRepetition ||
            RS == core::RepetitionStatus::SuperiorRepetition) {
            immediateUpdateByWin(LeafNode);
            PStat->incrementNumRepetition();
            // std::this_thread::yield();
            return isRunning();
        } else if (RS == core::RepetitionStatus::LossRepetition ||
                   RS == core::RepetitionStatus::InferiorRepetition) {
            immediateUpdateByLoss(LeafNode);
            PStat->incrementNumRepetition();
            // std::this_thread::yield();
            return isRunning();
        } else if (RS == core::RepetitionStatus::Repetition) {
            immediateUpdateByDraw(LeafNode,
                                  State->getSideToMove() == core::Black
                                      ? Config.BlackDrawValue
                                      : Config.WhiteDrawValue);
            PStat->incrementNumRepetition();
            // std::this_thread::yield();
            return isRunning();
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
                return isRunning();
            }
        }

        immediateUpdateByLoss(LeafNode);
        // std::this_thread::yield();
        return isRunning();
    }

    // This occurs when there is no available memory for edges.
    if (NumMoves == -1) {
        assert(LeafNode->getEdge() == nullptr);
        cancelVirtualLoss(LeafNode);
        PStat->incrementNumFailedToAllocateEdge();
        // std::this_thread::yield();
        return isRunning();
    }

    // Check delaration.
    if (State->canDeclare()) {
        immediateUpdateByWin(LeafNode);
        PStat->incrementNumCanDeclare();
        return isRunning();
    }

    // Check the number of plies.
    if (State->getPly() >= Config.MaxPly) {
        immediateUpdateByDraw(LeafNode, State->getSideToMove() == core::Black
                                            ? Config.BlackDrawValue
                                            : Config.WhiteDrawValue);
        PStat->incrementNumOverMaxPly();
        return isRunning();
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
            PStat->incrementNumSucceededToAddEvaluationQueue();
        } else {
            // Our MCTS implementation marks a node as “in expansion” for speed:
            //     (visit_count == 0 && virtual_loss == 1)
            // Because the leaf was already expanded and a virtual loss was
            // propagated from the root to here, we have to roll both back.
            //
            // IMPORTANT: release the edges *before* cancelling the virtual loss.
            // If we cancelled the virtual loss first, the node would momentarily have
            //     visit_count == 0 && virtual_loss == 0
            // while its edges were still present. Another thread could then reach
            // this leaf node and re-expand it, causing a data race.
            LeafNode->releaseEdges(EA);
            cancelVirtualLoss(LeafNode);
            PStat->incrementNumFailedToAddEvaluationQueue();
        }
    }

    return isRunning();
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
