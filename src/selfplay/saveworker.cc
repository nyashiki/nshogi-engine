//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "saveworker.h"

#include <cstdio>

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/statebuilder.h>
#include <nshogi/io/file.h>
#include <nshogi/io/sfen.h>
#include <nshogi/ml/simpleteacher.h>

namespace nshogi {
namespace engine {
namespace selfplay {

SaveWorker::SaveWorker(SelfplayInfo* SI, FrameQueue* SVQ, FrameQueue* SCQ,
                       std::size_t NumSelfplayGames, const char* SavePath, bool IgnoreDraw)
    : worker::Worker(true)
    , NumSelfplayGamesToStop(NumSelfplayGames)
    , SInfo(SI)
    , SaveQueue(SVQ)
    , SearchQueue(SCQ)
    , IgnoreDrawGames(IgnoreDraw) {

    Ofs.open(SavePath, std::ios::binary);
    if (!Ofs) {
        throw std::runtime_error(std::string("Failed to open ") + SavePath +
                                 ".");
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

        if (!IgnoreDrawGames || Task->getWinner() != core::NoColor) {
            updateStatistics(Task.get());
            save(Task.get());
        }

        assert(Task->getPhase() == SelfplayPhase::Save);
        if (SInfo->numGenerated() + SInfo->getNumOnGoinggames() < NumSelfplayGamesToStop) {
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
        SInfo->updateAveragePly(F->getState()->getPly(false));
    } else {
        SInfo->updateAveragePlyDraw(F->getState()->getPly(false));
    }

    if (F->getWinner() == core::Black) {
        SInfo->incrementBlackWin();
    } else if (F->getWinner() == core::White) {
        SInfo->incrementWhiteWin();
    } else {
        SInfo->incrementDraw();
    }

    if (F->getState()->canDeclare()) {
        SInfo->incrementDeclare();
    }

    LatestGame = io::sfen::stateToSfen(*F->getState());
}

void SaveWorker::printStatistics(bool Force) const {
    const auto CurrentTime = std::chrono::steady_clock::now();
    const auto ElapsedFromPrevious =
        std::chrono::duration_cast<std::chrono::milliseconds>(CurrentTime -
                                                              PreviousPrintTime)
            .count();

    if (!Force && ElapsedFromPrevious < 10000) {
        return;
    }

    PreviousPrintTime = CurrentTime;

    const auto Elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             CurrentTime - StartTime)
                             .count();
    const uint32_t ElapsedHour = (uint32_t)(Elapsed / 1000 / 60 / 60);
    const uint32_t ElapsedMinute = ((uint32_t)(Elapsed / 1000 / 60) % 60);
    const uint32_t ElapsedSecond = ((uint32_t)(Elapsed / 1000) % 60);

    std::printf("\x1b[2J\x1b[0;0H");
    std::printf("\n  Running selfplay ...\n\n");
    std::printf("    Elapsed:\n");
    std::printf("        %02u:%02u:%02u (%.3lf games per second)\n\n",
                ElapsedHour, ElapsedMinute, ElapsedSecond,
                (double)SInfo->numGenerated() / (double)Elapsed * 1000.0);
    std::printf("    Results:\n");
    std::printf("        %" PRIu64 " - %" PRIu64 " - %" PRIu64 "\n\n",
                    SInfo->numBlackWin(), SInfo->numDraw(), SInfo->numWhiteWin());
    std::printf("    Statistics:\n");
    std::printf("        - Black win rate: %.3lf\n", SInfo->blackWinRate());
    std::printf("        - White win rate: %.3lf\n", SInfo->whiteWinRate());
    std::printf("        - Draw rate: %.3lf\n", SInfo->drawRate());
    std::printf("        - Declare ratio: %.3lf\n", SInfo->declareRate());
    std::printf("        - Average ply: %.3lf\n", SInfo->averagePly());
    std::printf("        - Average ply (draw): %.3lf\n", SInfo->averagePlyDraw());
    std::printf("\n");
    std::printf("    Evaluation statisitcs:\n");
    std::printf("        - Average batch size: %.3lf\n",
                SInfo->getAverageBatchSize());
    std::printf("        - Evaluation cache hit ratio: %.3lf%%\n",
                SInfo->getCacheHitRatio() * 100.0);
    std::printf("\n");
    std::printf("    Latest game:\n");
    std::printf("        %s\n", LatestGame.c_str());
}

void SaveWorker::save(Frame* F) {
    ml::SimpleTeacher STeacher;
    core::State Replay =
        core::StateBuilder::newState(F->getState()->getInitialPosition());

    STeacher.setConfig(*F->getStateConfig());
    STeacher.setWinner(F->getWinner());

    while (Replay.getPly(false) < F->getState()->getPly(false)) {
        const auto NextMove =
            F->getState()->getHistoryMove(Replay.getPly(false));
        STeacher.setState(Replay);
        STeacher.setNextMove(core::Move16(NextMove));
        io::file::simple_teacher::save(Ofs, STeacher);

        Replay.doMove(F->getState()->getHistoryMove(Replay.getPly(false)));
    }
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
