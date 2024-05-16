#include "leafcollector.h"
#include "../evaluate/preset.h"

#include <limits>
#include <cmath>

#include <nshogi/core/movegenerator.h>

namespace nshogi {
namespace engine {
namespace mcts {

template <typename Features>
LeafCollector<Features>::LeafCollector(EvaluationQueue<Features>* EQ, MutexPool<lock::SpinLock>* MP, CheckmateSearcher* CS)
    : worker::Worker(true)
    , EQueue(EQ)
    , MtxPool(MP)
    , CSearcher(CS) {
    // Worker = std::thread(&LeafCollector<Features>::mainLoop, this);
}

template <typename Features>
LeafCollector<Features>::~LeafCollector<Features>() {
}

// template <typename Features>
// void LeafCollector<Features>::start() {
//     IsRunnning.store(true, std::memory_order_release);
//     CV.notify_one();
// }
//
// template <typename Features>
// void LeafCollector<Features>::stop() {
//     IsRunnning.store(false, std::memory_order_release);
//
//     while (IsThreadWorking.load(std::memory_order_acquire)) {
//         // Busy loop until the thread stops.
//         std::this_thread::yield();
//     }
// }
//
// template <typename Features>
// void LeafCollector<Features>::await() {
//     std::unique_lock<std::mutex> Lock(AwaitMutex);
//
//     AwaitCV.wait(Lock, [this]() {
//         return !IsRunnning.load(std::memory_order_relaxed)
//                 && !IsThreadWorking.load(std::memory_order_relaxed);
//     });
// }

template <typename Features>
void LeafCollector<Features>::updateRoot(const core::State& S, const core::StateConfig& StateConfig, Node* Root) {
    State = std::make_unique<core::State>(S.clone());
    Config = StateConfig;
    RootNode = Root;

    RootPly = State->getPly();
}

// template <typename Features>
// void LeafCollector<Features>::mainLoop() {
//     while (true) {
//         std::unique_lock<std::mutex> Lock(Mutex);
//
//         if (!IsRunnning.load(std::memory_order_relaxed)) {
//             IsThreadWorking.store(false, std::memory_order_relaxed);
//             AwaitCV.notify_all();
//         }
//
//         CV.wait(Lock, [this]() {
//             return IsRunnning.load(std::memory_order_relaxed)
//                     || IsExiting.load(std::memory_order_relaxed);
//         });
//
//         if (IsExiting.load(std::memory_order_relaxed)) {
//             return;
//         }
//
//         IsThreadWorking.store(true, std::memory_order_relaxed);
//
//         while (IsRunnning.load(std::memory_order_relaxed)) {
//
//             // std::this_thread::sleep_for(std::chrono::milliseconds(2));
//         }
//     }
// }

template <typename Features>
Node* LeafCollector<Features>::collectOneLeaf() {
    Node* CurrentNode = RootNode;

    while (true) {
        const uint64_t Visits = CurrentNode->getVisitsAndVirtualLoss() & Node::VisitMask;

        if (CSearcher != nullptr) {
            // If checkmate searcher is enabled and the node has not been
            // tried to solve, feed the node into the checkmate searcher.
            if (CurrentNode->getSolverResult().isNone()) {
                CSearcher->addTask(CurrentNode, State->getPosition());
            }
        }

        if (Visits == 0) {
            if (CurrentNode == RootNode) {
                if (MtxPool != nullptr) {
                    MtxPool->getRootMtx()->lock();
                }

                // After get the lock, the node might be already
                // extracted by another thread, so check its visit again.
                if (CurrentNode->getVisitsAndVirtualLoss() != 0ULL) {
                    if (MtxPool != nullptr) {
                        MtxPool->getRootMtx()->unlock();
                    }

                    // Since the node has been already extracted by another thread,
                    // this node is no longer a leaf node.
                    return nullptr;
                }

                CurrentNode->incrementVirtualLoss();

                if (MtxPool != nullptr) {
                    MtxPool->getRootMtx()->unlock();
                }

                return CurrentNode;
            }

            // Except for the root node, if the number of visits of the node is zero,
            // it means the node is being extracted so we return nullptr here.
            return nullptr;
        }

        const uint16_t NumChildren = CurrentNode->getNumChildren();

        // If the number of visits of the node is larger than zero and
        // the number of children of it is zero, the state is a terminal state
        // and therefore the node is a leaf node.
        if (Visits > 0 && NumChildren == 0) {
            break;
        }

        // If the state is a repetition state, we regard the node as a leaf node.
        // But when the node is the root node, we don't so that the search can proceed
        // to have the children of the root in the tree.
        if (CurrentNode != RootNode &&
                CurrentNode->getRepetitionStatus() != core::RepetitionStatus::NoRepetition) {
            break;
        }

        // If `getPlyToTerminalSolved()` returns a non-zero value,
        // it means the game theoretical value has already been solved and therefore
        // we don't have to search this node furthermore.
        if (CurrentNode->getPlyToTerminalSolved() != 0) {
            break;
        }

        Edge* E = computeUCBMaxEdge(CurrentNode, NumChildren, false);
        // computeUCBMaxEdge() can return nullptr if many threads reaches on the same leaf node.
        if (E == nullptr) {
            return nullptr;
        }

        State->doMove(State->getMove32FromMove16(E->getMove()));

        Node* Target = E->getTarget();
        if (Target == nullptr) {
            // If `Target` is nullptr, we have not extracted the child
            // of this node after transitioning by the edge.

            lock::SpinLock* EdgeMtx = nullptr;
            if (MtxPool != nullptr) {
                EdgeMtx = MtxPool->get(reinterpret_cast<void*>(E));
                EdgeMtx->lock();

                if (E->getTarget() != nullptr) {
                    // This thread has reached a leaf node but another thread also
                    // had reached this leaf node and has evaluated this leaf node.
                    // Therefore E->getTarget() is no longer nullptr and
                    // nothing to do is left.
                    EdgeMtx->unlock();
                    return nullptr;
                }
            }

            auto NewNode = std::make_unique<Node>(CurrentNode);
            auto* NewNodePtr = NewNode.get();
            incrementVirtualLosses(NewNodePtr);
            E->setTarget(std::move(NewNode));

            if (EdgeMtx != nullptr) {
                EdgeMtx->unlock();
            }

            return NewNodePtr;
        }

        CurrentNode = E->getTarget();
    }

    incrementVirtualLosses(CurrentNode);
    return CurrentNode;
}

template <typename Feature>
bool LeafCollector<Feature>::expandLeaf(Node* LeafNode) {
    const auto Moves = core::MoveGenerator::generateLegalMoves(*State);

    if (Moves.size() == 0) {
        return false;
    }

    LeafNode->expand(Moves);
    return true;
}

template <typename Feature>
void LeafCollector<Feature>::immediateUpdateByWin(Node* LeafNode) {
    LeafNode->setEvaluation(nullptr, 1.0f, 0.0f);
    LeafNode->setPlyToTerminalSolved(1);
    LeafNode->updateAncestors(1.0f, 0.0f);
}

template <typename Feature>
void LeafCollector<Feature>::immediateUpdateByLoss(Node* LeafNode) {
    LeafNode->setEvaluation(nullptr, 0.0f, 0.0f);
    LeafNode->setPlyToTerminalSolved(-1);
    LeafNode->updateAncestors(0.0f, 0.0f);
}

template <typename Feature>
void LeafCollector<Feature>::immediateUpdateByDraw(Node* LeafNode) {
    LeafNode->setEvaluation(nullptr, 0.5f, 1.0f);
    LeafNode->updateAncestors(0.5f, 1.0f);
}

template <typename Feature>
void LeafCollector<Feature>::immediateUpdate(Node* LeafNode) {
    float WinRate = LeafNode->getWinRatePredicted();
    float DrawRate = LeafNode->getDrawRatePredicted();

    const auto RS = LeafNode->getRepetitionStatus();

    if (RS == core::RepetitionStatus::WinRepetition
            || RS == core::RepetitionStatus::SuperiorRepetition) {
        WinRate = 1.0;
        DrawRate = 0.0;
    } else if (RS == core::RepetitionStatus::LossRepetition
            || RS == core::RepetitionStatus::InferiorRepetition) {
        WinRate = 0.0;
        DrawRate = 0.0;
    } else if (RS == core::RepetitionStatus::Repetition) {
        WinRate = 0.5;
        DrawRate = 1.0;
    } else {
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

template <typename Features>
Edge* LeafCollector<Features>::computeUCBMaxEdge(Node* N, uint16_t NumChildren, bool regardNotVisitedWin) {
    assert(NumChildren > 0);
    const uint64_t CurrentVisitsAndVirtualLoss = N->getVisitsAndVirtualLoss();
    const uint64_t CurrentVisits = CurrentVisitsAndVirtualLoss & Node::VisitMask;
    const uint64_t CurrentVirtualLoss = CurrentVisitsAndVirtualLoss >> Node::VirtualLossShift;

    if (CurrentVisits == 1) {
        // If the number of visit is equal to one, it means all children
        // is not expanded yet. Recall the UCB fomular, the most promising edge is
        // the edge with the highest prior. Since we have sorted the children along
        // their prior, we can simply select 0-th edge if the virtual loss is zero.
        // If the virtual loss is not zero, we simply choose `virtual-loss`-th element.
        if (CurrentVirtualLoss < NumChildren) {
            return N->getEdge(CurrentVirtualLoss);
        } else {
            // When the virtual loss is larger than or equal to the number of children,
            // all children will be extracted so nothing to do here.
            return nullptr;
        }
    }

    const uint64_t CurrentVirtualVisits = CurrentVisits + CurrentVirtualLoss;

    const double Const = std::log((double)(CurrentVirtualVisits + CBase) / (double)CBase + CInit) *
                        std::sqrt((double)CurrentVirtualVisits);

    Edge* UCBMaxEdge = nullptr;
    double UCBMaxValue = std::numeric_limits<double>::lowest();
    bool IsAllTargetLoss = true;
    int16_t WinTargetPlyMin = 1024;
    int16_t LossTargetPlyMax = 0;

    Edge* ShortestWinEdge = nullptr;
    Edge* LongestLossEdge = nullptr;

    for (uint16_t I = 0; I < NumChildren; ++I) {
        auto* const Edge = N->getEdge(I);
        auto* const Child = Edge->getTarget();

        // The child is not visited yet.
        if (Child == nullptr) {
            const double UCBValue = regardNotVisitedWin
                ? (1.0 + Const * Edge->getProbability())
                : (Const * Edge->getProbability());

            // Since there is at least one unvisited child, which means
            // the child is not solved, we don't know all children are loss states.
            IsAllTargetLoss = false;

            if (UCBValue > UCBMaxValue) {
                UCBMaxValue = UCBValue;
                UCBMaxEdge = Edge;
            }
            continue;
        }

        const uint64_t ChildVisitsAndVirtualLoss = Child->getVisitsAndVirtualLoss();
        const uint64_t ChildVisits = ChildVisitsAndVirtualLoss & Node::VisitMask;
        const uint64_t ChildVirtualLoss = ChildVisitsAndVirtualLoss >> Node::VirtualLossShift;

        if (ChildVisits == 0 && ChildVirtualLoss > 0) {
            // When the number of visits is zero and the number of
            // virtual loss is larger than zero of a node,
            // it means the node is being extracted.
            IsAllTargetLoss = false;
            continue;
        }

        const uint64_t ChildVirtualVisits = ChildVisits + ChildVirtualLoss;

        const int16_t PlyToTerminal = Child->getPlyToTerminalSolved();
        if (PlyToTerminal > 0) {
            // If `PlyToTerminal` of a child is larger than zero,
            // it means the child is a win state from the perspective of
            // the side to move of the child state. Therefore, if we choose
            // this child, we will lose the game. When losing, one of the
            // best moves is a move that has longest sequence of moves to a terminal.
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
            // best moves is a move that has shortest sequence of moves to a terminal.
            if (NegativePlyToTerminal < WinTargetPlyMin) {
                ShortestWinEdge = Edge;
                WinTargetPlyMin = NegativePlyToTerminal;
            }
            continue;
        }

        // Since we have already at least one winning edge,
        // if the current searching edge is not a winning edge, we don't have to
        // search it so `continue` here.
        if (ShortestWinEdge != nullptr) {
            continue;
        }

        const double ChildWinRate = computeWinRateOfChild(Child, ChildVisits, ChildVirtualVisits);
        const double UCBValue = ChildWinRate + Const * Edge->getProbability() / ((double)(1 + ChildVirtualVisits));

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

template <typename Features>
double LeafCollector<Features>::computeWinRateOfChild(Node* Child, uint64_t ChildVisits, uint64_t ChildVirtualVisits) {
    const double ChildWinRateAccumulated = Child->getWinRateAccumulated();
    const double ChildDrawRateAcuumulated = Child->getDrawRateAccumulated();

    const double WinRate = ((double)ChildVisits - ChildWinRateAccumulated) / (double)ChildVirtualVisits;
    const double DrawRate = ChildDrawRateAcuumulated / (double)ChildVirtualVisits;

    const double DrawValue = (State->getSideToMove() == core::Black)
                                ? Config.BlackDrawValue
                                : Config.WhiteDrawValue;

    return DrawRate * DrawValue + (1.0 - DrawRate) * WinRate;
}

template <typename Features>
void LeafCollector<Features>::incrementVirtualLosses(Node* N) {
    do {
        N->incrementVirtualLoss();
        N = N->getParent();
    } while (N != nullptr);
}

template <typename Features>
bool LeafCollector<Features>::doTask() {
    // Go back to the root state.
    while (State->getPly() != RootPly) {
        State->undoMove();
    }

    Node* LeafNode = collectOneLeaf();

    if (LeafNode != nullptr) {
        const uint64_t NumVisitsAndVirtualLoss = LeafNode->getVisitsAndVirtualLoss();
        const uint64_t NumVisits = NumVisitsAndVirtualLoss & Node::VisitMask;
        const uint64_t VirtualLoss = NumVisitsAndVirtualLoss >> Node::VirtualLossShift;

        if (NumVisits > 0) {
            immediateUpdate(LeafNode);
        } else if (VirtualLoss == 1) {
            if (LeafNode != RootNode) {
                const auto RS = State->getRepetitionStatus();
                LeafNode->setRepetitionStatus(RS);
                if (RS == core::RepetitionStatus::WinRepetition
                        || RS == core::RepetitionStatus::SuperiorRepetition) {
                    immediateUpdateByWin(LeafNode);
                    return false;
                } else if (RS == core::RepetitionStatus::LossRepetition
                        || RS == core::RepetitionStatus::InferiorRepetition) {
                    immediateUpdateByLoss(LeafNode);
                    return false;
                } else if (RS == core::RepetitionStatus::Repetition) {
                    immediateUpdateByDraw(LeafNode);
                    return false;
                }
            }

            if (expandLeaf(LeafNode)) {
                EQueue->add(*State, Config, LeafNode);
            } else {
                bool IsCheckmatedByPawn = false;
                if (State->getPly() > 0) {
                    const auto LastMove = State->getLastMove();
                    if (LastMove.drop() && LastMove.pieceType() == core::PTK_Pawn) {
                        immediateUpdateByWin(LeafNode);
                        IsCheckmatedByPawn = true;
                    }
                }

                if (!IsCheckmatedByPawn) {
                    immediateUpdateByLoss(LeafNode);
                }
            }
        }
    }

    return false;
}

template class LeafCollector<evaluate::preset::SimpleFeatures>;
template class LeafCollector<evaluate::preset::CustomFeaturesV1>;

} // namespace mcts
} // namespace engine
} // namespace nshogi
