//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "../allocator/fixed_allocator.h"
#include "../allocator/segregated_free_list.h"
#include "../argparser.h"
#include "evaluationworker.h"
#include "framequeue.h"
#include "saveworker.h"
#include "selfplayinfo.h"
#include "worker.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <nshogi/core/initializer.h>
#include <nshogi/io/sfen.h>

int main(int Argc, char* Argv[]) {
    nshogi::engine::ArgParser Parser;

    Parser.addOption("model", "Path to the onnx file.");
    Parser.addOption("batch-size", "128", "Batch size.");
    Parser.addOption("frame-pool-size", "1024", "Frame pool size.");
    Parser.addOption("num-gpus", "1", "The number of gpus.");
    Parser.addOption("num-search-workers", "1",
                     "The number of search workers.");
    Parser.addOption("num-evaluation-workers-per-gpu", "1",
                     "The number of evaluation workers per gpu.");
    Parser.addOption("memory-size", "1024", "Memory size (MB).");
    Parser.addOption("evaluation-cache-memory-size", "1024",
                     "Evaluation cache memory size (MB).");
    Parser.addOption("num-selfplay-games", "500000",
                     "The number of selfplay games.");
    Parser.addOption('o', "out", "out.bin", "The output teacher file.");
    Parser.addOption("initial-positions", "",
                     "Sfen file that contains sfen positions.");
    Parser.addOption("use-shogi816k", "Use shogi816k positions.");

    Parser.parse(Argc, Argv);

    if (Parser.isSpecified("help")) {
        Parser.showHelp();
        return 0;
    }

    using namespace nshogi;
    using namespace nshogi::engine;
    using namespace nshogi::engine::selfplay;

    core::initializer::initializeAll();

    // Setup allocator.
    auto NodeAllocator =
        std::make_unique<allocator::FixedAllocator<sizeof(mcts::Node)>>();
    auto EdgeAllocator =
        std::make_unique<allocator::SegregatedFreeListAllocator>();

    const std::size_t AVAILABLE_MEMORY_MB =
        (std::size_t)std::stoull(Parser.getOption("memory-size"));
    NodeAllocator->resize(
        (std::size_t)(0.1 * (double)AVAILABLE_MEMORY_MB * 1024ULL * 1024ULL));
    EdgeAllocator->resize(
        (std::size_t)(0.9 * (double)AVAILABLE_MEMORY_MB * 1024ULL * 1024ULL));

    // Prepare queue.
    auto SearchQueue = std::make_unique<FrameQueue>();
    auto EvaluationQueue = std::make_unique<FrameQueue>();
    auto SaveQueue = std::make_unique<FrameQueue>();

    // Prepare garbage collectors.
    auto GC = std::make_unique<mcts::GarbageCollector>(1, NodeAllocator.get(),
                                                       EdgeAllocator.get());

    // Prepare evaluation cache.
    const std::size_t EVALCACHE_MEMORY_MB = (std::size_t)std::stoull(
        Parser.getOption("evaluation-cache-memory-size"));
    ;
    auto EvalCache = std::make_unique<mcts::EvalCache>(EVALCACHE_MEMORY_MB);

    // Prepare empty frames.
    const std::size_t NUM_FRAME_POOL =
        (std::size_t)std::stoull(Parser.getOption("frame-pool-size"));
    for (std::size_t I = 0; I < NUM_FRAME_POOL; ++I) {
        auto F = std::make_unique<Frame>(GC.get(), NodeAllocator.get());
        F->setEvaluationCache(EvalCache.get());
        SearchQueue->add(std::move(F));
    }

    auto SInfo = std::make_unique<SelfplayInfo>(NUM_FRAME_POOL);

    // Prepare initial positions.
    std::unique_ptr<std::vector<core::Position>> InitialPositions;
    const bool USE_SHOGI816K = Parser.isSpecified("use-shogi816k");
    {
        const std::string INITIAL_POSITIONS_PATH =
            Parser.getOption("initial-positions");

        if (INITIAL_POSITIONS_PATH != "" && USE_SHOGI816K) {
            throw std::runtime_error("You can't specify --initial-positions "
                                     "and --use-shogi816k at the same time.");
        }

        std::ifstream Ifs(INITIAL_POSITIONS_PATH);
        if (INITIAL_POSITIONS_PATH != "" && !Ifs) {
            throw std::runtime_error("intiial positions option was specified "
                                     "but failed to open the file.");
        }
        if (Ifs) {
            InitialPositions = std::make_unique<std::vector<core::Position>>();

            std::string Line;
            while (std::getline(Ifs, Line)) {
                if (Line == "" || Line[0] == '#') {
                    continue;
                }
                InitialPositions->emplace_back(
                    nshogi::io::sfen::PositionBuilder::newPosition(Line));
            }
        }
    }

    // Prepare workers.
    const std::size_t NUM_SEARCH_WORKERS =
        (std::size_t)std::stoull(Parser.getOption("num-search-workers"));
    std::vector<std::unique_ptr<worker::Worker>> SearchWorkers;
    for (std::size_t I = 0; I < NUM_SEARCH_WORKERS; ++I) {
        SearchWorkers.emplace_back(std::make_unique<Worker>(
            SearchQueue.get(), EvaluationQueue.get(), SaveQueue.get(),
            NodeAllocator.get(), EdgeAllocator.get(), EvalCache.get(),
            InitialPositions.get(), USE_SHOGI816K, SInfo.get()));
    }

    const std::size_t NUM_EVALUATION_WORKERS_PER_GPU = (std::size_t)std::stoull(
        Parser.getOption("num-evaluation-workers-per-gpu"));
    const std::size_t NUM_GPUS =
        (std::size_t)std::stoull(Parser.getOption("num-gpus"));
    const std::size_t BATCH_SIZE =
        (std::size_t)std::stoull(Parser.getOption("batch-size"));
    const std::string WEIGHT_PATH = Parser.getOption("model");
    std::vector<std::unique_ptr<worker::Worker>> EvaluationWorkers;
    for (std::size_t I = 0; I < NUM_GPUS; ++I) {
        for (std::size_t J = 0; J < NUM_EVALUATION_WORKERS_PER_GPU; ++J) {
            EvaluationWorkers.emplace_back(std::make_unique<EvaluationWorker>(
                EvaluationWorkers.size(), I, BATCH_SIZE, WEIGHT_PATH.c_str(),
                EvaluationQueue.get(), SearchQueue.get(), SInfo.get()));
        }
    }

    const std::size_t NUM_SELFPLAY_GAMES =
        (std::size_t)std::stoull(Parser.getOption("num-selfplay-games"));
    const std::string SAVE_PATH = Parser.getOption("out");
    auto Saver = std::make_unique<SaveWorker>(
        SInfo.get(), SaveQueue.get(), SearchQueue.get(), NUM_SELFPLAY_GAMES,
        SAVE_PATH.c_str());

    // Launch workers.
    Saver->start();
    for (auto& Worker : EvaluationWorkers) {
        Worker->start();
    }
    for (auto& Worker : SearchWorkers) {
        Worker->start();
    }

    SInfo->waitUntilAllGamesFinished();

    SearchQueue->close();
    EvaluationQueue->close();
    SaveQueue->close();

    // Wait workers.
    for (auto& Worker : SearchWorkers) {
        Worker->stop();
        Worker->await();
    }
    for (auto& Worker : EvaluationWorkers) {
        Worker->stop();
        Worker->await();
    }
    Saver->stop();
    Saver->await();

    std::cout << "Selfplay finished." << std::endl;

    return 0;
}
