#include "watchdog.h"
#include "../globalconfig.h"

#include <chrono>
#include <cmath>

namespace nshogi {
namespace engine {
namespace mcts {

Watchdog::Watchdog(std::shared_ptr<logger::Logger> Logger)
    : worker::Worker(false)
    , StopSearchingCallback(nullptr)
    , PLogger(std::move(Logger)) {
}

Watchdog::~Watchdog() {
}

void Watchdog::updateRoot(const core::State* S, const core::StateConfig* SC, Node* N) {
    State = S;
    Config = SC;
    Root = N;
}

void Watchdog::setLimit(const engine::Limit& Lim) {
    Limit = std::make_unique<engine::Limit>(Lim);
}

void Watchdog::setStopSearchingCallback(std::function<void()> Callback) {
    StopSearchingCallback = Callback;
}

bool Watchdog::doTask() {
    const auto StartTime = std::chrono::steady_clock::now();
    auto LogTimePrevious = StartTime;
    const uint64_t NumNodesAtStarted = Root->getVisitsAndVirtualLoss() & Node::VisitMask;

    BestEdgePrevious = nullptr;

    while (true) {
        const auto CurrentTime = std::chrono::steady_clock::now();
        const uint32_t Elapsed =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>
                    (CurrentTime - StartTime).count());
        const uint32_t LogElapsed =
            static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>
                    (CurrentTime - LogTimePrevious).count());

        if (LogElapsed >= GlobalConfig::getConfig().getLogMargin()) {
            dumpPVLog(NumNodesAtStarted, Elapsed);
            LogTimePrevious = CurrentTime;
        }

        if (isRootSolved()) {
            break;
        }

        if (!getIsRunning()) {
            break;
        }

        if (!Limit->isNoLimit()) {
            if (checkResourceBudget()) {
                break;
            }

            if (checkThinkingTimeBudget(Elapsed)) {
                break;
            }

            if (hasMadeUpMind(Elapsed)) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cerr << "[doWatchdogWork()] stop workers ..." << std::endl;
    if (StopSearchingCallback != nullptr) {
        std::cerr << "[doWatchdogWork()] calling callback ..." << std::endl;
        StopSearchingCallback();
        std::cerr << "[doWatchdogWork()] calling callback ... ok." << std::endl;
    }
    std::cerr << "[doWatchdogWork()] stop workers ... ok." << std::endl;
    const auto CurrentTime = std::chrono::steady_clock::now();
    const uint32_t Elapsed =
        static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>
                (CurrentTime - StartTime).count());
    dumpPVLog(NumNodesAtStarted, Elapsed);

    return false;
}

bool Watchdog::isRootSolved() const {
    return Root->getPlyToTerminalSolved() != 0;
}

bool Watchdog::checkResourceBudget() const {
    const auto& NodeAllocator = allocator::getNodeAllocator();
    const double Factor = GlobalConfig::getConfig().getMemoryLimitFactor();

    if (NodeAllocator.getTotal() > 0 &&
            (double)NodeAllocator.getUsed() > (double)NodeAllocator.getTotal() * Factor) {
        PLogger->printLog("Memory limit (Node).");
        return true;
    }

    const auto& EdgeAllocator = allocator::getEdgeAllocator();
    if (EdgeAllocator.getTotal() > 0 &&
            (double)EdgeAllocator.getUsed() > (double)EdgeAllocator.getTotal() * Factor) {
        PLogger->printLog("Memory limit (Edge).");
        return true;
    }

    return false;
}

bool Watchdog::checkThinkingTimeBudget(uint32_t Elapsed) const {
    if (Limit->isNoLimit()) {
        return false;
    }

    const uint32_t Budget = Limit->TimeLimitMilliSeconds
        + Limit->ByoyomiMilliSeconds
        + Limit->IncreaseMilliSeconds;

    return Elapsed + GlobalConfig::getConfig().getThinkingTimeMargin() >= Budget;
}

bool Watchdog::hasMadeUpMind(uint32_t Elapsed) {
    if (Root == nullptr) {
        return false;
    }

    if (Elapsed < ElapsedPrevious + 470) {
        return false;
    }

    const uint16_t NumChildren = Root->getNumChildren();

    uint64_t SumVisits = 0;

    std::vector<double> Visits(NumChildren, 0.0);
    for (uint16_t I = 0; I < NumChildren; ++I) {
        Edge* E = Root->getEdge(I);
        Node* Child = E->getTarget();

        if (Child != nullptr) {
            const uint64_t V = Child->getVisitsAndVirtualLoss() & Node::VisitMask;
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

    Edge* BestEdge = Root->mostPromisingEdge();

    if (BestEdge == BestEdgePrevious && Visits.size() == VisitsPrevious.size()) {
        double KLDivergence = 0.0;
        double KLDivergenceToPredicted = 0.0;

        for (std::size_t I = 0; I < Visits.size(); ++I) {
            if (VisitsPrevious[I] == 0.0) {
                continue;
            } else if (Visits[I] == 0.0) {
                KLDivergence = std::numeric_limits<double>::max();
                break;
            }

            const double Predicted = Root->getEdge(I)->getProbability();
            const double KLD = VisitsPrevious[I] * std::log(VisitsPrevious[I] / Visits[I]);

            KLDivergence += KLD;

            if (Predicted > 0) {
                const double KLDToP = Predicted * std::log(Predicted / Visits[I]);
                KLDivergenceToPredicted += KLDToP;
            }
        }

        const double KLDThreshold =
            (KLDivergenceToPredicted < 0.4) ? 1e-5 : 1e-6;

        if (KLDivergence < KLDThreshold) {
            return true;
        }
    }

    ElapsedPrevious = Elapsed;
    BestEdgePrevious = BestEdge;
    VisitsPrevious.swap(Visits);

    return false;
}

void Watchdog::dumpPVLog(uint64_t NumNodesAtStarted, uint32_t Elapsed) const {
    logger::PVLog Log = getPVLog();

    Log.ElapsedMilliSeconds = Elapsed;
    if (Elapsed > 0) {
        Log.NodesPerSecond = (uint64_t)(
                (double)(Log.NumNodes - NumNodesAtStarted) * 1000ULL / (double)Elapsed);
    }

    PLogger->printPVLog(Log);
}

logger::PVLog Watchdog::getPVLog() const {
    logger::PVLog Log;

    Node* N = Root;

    const uint64_t Visits = Root->getVisitsAndVirtualLoss() & Node::VisitMask;
    if (Visits == 0) {
        return Log;
    }

    Log.NumNodes = Visits;
    Log.SolvedGameEndPly = N->getPlyToTerminalSolved();
    Log.WinRate = N->getWinRateAccumulated() / (double)Visits;
    Log.DrawRate = N->getDrawRateAccumulated() / (double)Visits;
    Log.DrawValue = (State->getSideToMove() == core::Black)
                ? Config->BlackDrawValue : Config->WhiteDrawValue;

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

} // namespace mcts
} // namespace engine
} // namespace nshogi
