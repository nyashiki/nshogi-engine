#include "saveworker.h"

#include <cstdio>

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/sfen.h>
#include <nshogi/ml/simpleteacher.h>
#include <nshogi/io/file.h>

namespace nshogi {
namespace engine {
namespace selfplay {

SaveWorker::SaveWorker(SelfplayInfo* SI, FrameQueue* SVQ, FrameQueue* SCQ, std::size_t NumSelfplayGames, const char* SavePath)
    : worker::Worker(true)
    , NumSelfplayGamesToStop(NumSelfplayGames)
    , SInfo(SI)
    , SaveQueue(SVQ)
    , SearchQueue(SCQ) {

    Ofs.open(SavePath, std::ios::binary);
    if (!Ofs) {
        throw std::runtime_error(std::string("Failed to open ") + SavePath + ".");
    }

    StartTime = std::chrono::steady_clock::now();
    PreviousPrintTime = StartTime;

    spawnThread();
}

bool SaveWorker::doTask() {
    TasksToAdd.clear();

    auto Tasks = SaveQueue->getAll();

    while (!Tasks.empty()) {
        auto Task = std::move(Tasks.front());
        Tasks.pop();

        updateStatistics(Task.get());
        save(Task.get());

        assert(Task->getPhase() == SelfplayPhase::Save);
        if (Statistics.NumBlackWin + Statistics.NumDraw + Statistics.NumWhiteWin + SInfo->getNumOnGoinggames() < NumSelfplayGamesToStop) {
            Task->setPhase(SelfplayPhase::Initialization);
            TasksToAdd.emplace_back(std::move(Task));
            printStatistics(false);
        } else {
            Task.reset(nullptr);
            SInfo->decrementNumOnGoingGames();
            printStatistics(true);
        }
    }

    if (TasksToAdd.size() > 0) {
        SearchQueue->add(TasksToAdd);
    }

    return false;
}

void SaveWorker::updateStatistics(Frame* F) {
    if (F->getWinner() != core::NoColor) {
        const uint64_t N = Statistics.NumBlackWin + Statistics.NumWhiteWin;
        Statistics.AveragePly =
            (Statistics.AveragePly * (double)N + (double)F->getState()->getPly(false)) / (double)(1 + N);
    } else {
        Statistics.AveragePlyDraw =
            (Statistics.AveragePlyDraw * (double)Statistics.NumDraw + (double)F->getState()->getPly(false))
            / (double)(1 + Statistics.NumDraw);
    }

    if (F->getWinner() == core::Black) {
        ++Statistics.NumBlackWin;
    } else if (F->getWinner() == core::White) {
        ++Statistics.NumWhiteWin;
    } else {
        ++Statistics.NumDraw;
    }

    if (F->getState()->canDeclare()) {
        ++Statistics.NumDeclare;
    }

    LatestGame = io::sfen::stateToSfen(*F->getState());
}

void SaveWorker::printStatistics(bool Force) const {
    const auto CurrentTime = std::chrono::steady_clock::now();
    const auto ElapsedFromPrevious = std::chrono::duration_cast<std::chrono::milliseconds>
        (CurrentTime - PreviousPrintTime).count();

    if (!Force && ElapsedFromPrevious < 10000) {
        return;
    }

    PreviousPrintTime = CurrentTime;

    const auto Elapsed = std::chrono::duration_cast<std::chrono::milliseconds>
        (CurrentTime - StartTime).count();
    const uint32_t ElapsedHour = (uint32_t)(Elapsed / 1000 / 60 / 60);
    const uint32_t ElapsedMinute = ((uint32_t)(Elapsed / 1000 / 60) % 60);
    const uint32_t ElapsedSecond = ((uint32_t)(Elapsed / 1000) % 60);

    const uint64_t NumMatches = Statistics.NumBlackWin + Statistics.NumDraw + Statistics.NumWhiteWin;
    const double BlackWinRate = (NumMatches == 0)
        ? 0.0 : ((double)Statistics.NumBlackWin / (double)NumMatches);
    const double DrawRate = (NumMatches == 0)
        ? 0.0 : ((double)Statistics.NumDraw / (double)NumMatches);
    const double WhiteWinRate = (NumMatches == 0)
        ? 0.0 : ((double)Statistics.NumWhiteWin / (double)NumMatches);
    const double DeclareRatio = (NumMatches == 0)
        ? 0.0 : ((double)Statistics.NumDeclare / (double)NumMatches);
    std::printf("\x1b[2J\x1b[0;0H");
    std::printf("\n  Running selfplay ...\n\n");
    std::printf("    Elapsed:\n");
    std::printf("        %02u:%02u:%02u (%.3lf games per second)\n\n",
            ElapsedHour, ElapsedMinute, ElapsedSecond, (double)NumMatches / (double)Elapsed * 1000.0);
    std::printf("    Results:\n");
    std::printf("        %" PRIu64 " - %" PRIu64 " - %" PRIu64 "\n\n",
            Statistics.NumBlackWin, Statistics.NumDraw, Statistics.NumWhiteWin);
    std::printf("    Statistics:\n");
    std::printf("        - Black win rate: %.3lf\n", BlackWinRate);
    std::printf("        - White win rate: %.3lf\n", WhiteWinRate);
    std::printf("        - Draw rate: %.3lf\n", DrawRate);
    std::printf("        - Declare ratio: %.3lf\n", DeclareRatio);
    std::printf("        - Average ply: %.3lf\n", Statistics.AveragePly);
    std::printf("        - Average ply (draw): %.3lf\n", Statistics.AveragePlyDraw);
    std::printf("\n");
    std::printf("    Evaluation statisitcs:\n");
    std::printf("        - Average batch size: %.3lf\n", SInfo->getAverageBatchSize());
    std::printf("        - Evaluation cache hit ratio: %.3lf%%\n", SInfo->getCacheHitRatio() * 100.0);
    std::printf("\n");
    std::printf("    Latest game:\n");
    std::printf("        %s\n", LatestGame.c_str());
}

void SaveWorker::save(Frame* F) {
    ml::SimpleTeacher STeacher;
    core::State Replay = core::StateBuilder::newState(F->getState()->getInitialPosition());

    STeacher.setConfig(*F->getStateConfig());
    STeacher.setWinner(F->getWinner());

    while (Replay.getPly(false) < F->getState()->getPly(false)) {
        const auto NextMove = F->getState()->getHistoryMove(Replay.getPly(false));
        STeacher.setState(Replay);
        STeacher.setNextMove(NextMove);
        io::file::simple_teacher::save(Ofs, STeacher);

        Replay.doMove(F->getState()->getHistoryMove(Replay.getPly(false)));
    }
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
