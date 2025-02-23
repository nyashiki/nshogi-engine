//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "usi.h"

#include "../command/executor.h"
#include "../book/bookmaker.h"
#include "usilogger.h"
#include "usioption.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

#include <nshogi/core/movegenerator.h>
#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/core/types.h>
#include <nshogi/io/sfen.h>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

namespace {

USIOption Option;
std::unique_ptr<command::Executor> Executor;
std::shared_ptr<USILogger> Logger = std::make_shared<USILogger>();

} // namespace

namespace {

constexpr static const char* USI_OPTION_MAX_PLY = "USI_MaxPly";
constexpr static const char* USI_OPTION_HASH = "USI_Hash";
constexpr static const char* USI_OPTION_PONDER = "USI_Ponder";
constexpr static const char* USI_OPTION_NUM_GPUS = "NumGPUs";
constexpr static const char* USI_OPTION_NUM_SEARCH_THREADS = "NumSearchThreads";
constexpr static const char* USI_OPTION_NUM_EVALUATION_THREADS_PER_GPU =
    "NumEvaluationThreadsPerGPU";
constexpr static const char* USI_OPTION_NUM_CHECKMATE_THREADS =
    "NumCheckmateSearchThreads";
constexpr static const char* USI_OPTION_BATCH_SIZE = "BatchSize";
constexpr static const char* USI_OPTION_BOOK_ENABLED = "IsBookEnabled";
constexpr static const char* USI_OPTION_WEIGHT_PATH = "WeightPath";
constexpr static const char* USI_OPTION_BOOK_PATH = "BookPath";
constexpr static const char* USI_OPTION_EVAL_CACHE_MEMORY_MB =
    "EvalCacheMemoryMB";
constexpr static const char* USI_OPTION_THINKING_TIME_MARGIN =
    "ThinkingTimeMargin";
constexpr static const char* USI_OPTION_BLACK_DRAW_VALUE = "BlackDrawValue";
constexpr static const char* USI_OPTION_WHITE_DRAW_VALUE = "WhiteDrawValue";
constexpr static const char* USI_OPTION_REPETITION_BOOK_ALLOWED =
    "RepetitionBookAllowed";

void setupOption(const Context* C) {
    Option.addIntOption(USI_OPTION_MAX_PLY, 320, 1, 99999);
    Option.addIntOption(USI_OPTION_HASH, (int64_t)C->getAvailableMemoryMB(),
                        1024LL, 1024 * 1024LL);
    Option.addBoolOption(USI_OPTION_PONDER, C->getPonderingEnabled());
    Option.addIntOption(USI_OPTION_NUM_GPUS, (int64_t)C->getNumGPUs(), 1, 16);
    Option.addIntOption(USI_OPTION_NUM_SEARCH_THREADS,
                        (int64_t)C->getNumSearchThreads(), 1, 2048);
    Option.addIntOption(USI_OPTION_NUM_EVALUATION_THREADS_PER_GPU,
                        (int64_t)C->getNumEvaluationThreadsPerGPU(), 1, 2048);
    Option.addIntOption(USI_OPTION_NUM_CHECKMATE_THREADS,
                        (int64_t)C->getNumCheckmateSearchThreads(), 0, 128);
    Option.addIntOption(USI_OPTION_BATCH_SIZE, (int64_t)C->getBatchSize(), 1,
                        4096);
    Option.addBoolOption(USI_OPTION_BOOK_ENABLED, C->isBookEnabled());
    Option.addFileNameOption(USI_OPTION_WEIGHT_PATH,
                             C->getWeightPath().c_str());
    Option.addFileNameOption(USI_OPTION_BOOK_PATH, C->getBookPath().c_str());
    Option.addIntOption(USI_OPTION_EVAL_CACHE_MEMORY_MB,
                        (int64_t)C->getEvalCacheMemoryMB(), 0LL, 1024 * 1024LL);
    Option.addIntOption(USI_OPTION_THINKING_TIME_MARGIN,
                        (int64_t)C->getThinkingTimeMargin(), 0LL, 60 * 1000);
    Option.addIntOption(USI_OPTION_BLACK_DRAW_VALUE,
                        (int)(C->getBlackDrawValue() * 100.0f), 0, 100);
    Option.addIntOption(USI_OPTION_WHITE_DRAW_VALUE,
                        (int)(C->getWhiteDrawValue() * 100.0f), 0, 100);
    Option.addBoolOption(USI_OPTION_REPETITION_BOOK_ALLOWED,
                         C->isRepetitionBookAllowed());
}

void showOption() {
    Option.showOption();
}

void greet() {
    Executor = std::make_unique<command::Executor>(Logger);
    setupOption(Executor->getContext());

    Logger->printRawMessage("id name ", USIName);
    Logger->printRawMessage("id author ", USIAuthor);

    showOption();

    Logger->printRawMessage("usiok");
}

void isready() {
    static bool IsFirstCall = true;

    if (!IsFirstCall) {
        Logger->printRawMessage("readyok");
        return;
    }

    IsFirstCall = false;

    using namespace command::commands;

    Executor->pushCommand(std::make_shared<BoolConfig>(
        Configurable::PonderEnabled, Option.getIntOption(USI_OPTION_PONDER)));
    Executor->pushCommand(std::make_shared<BoolConfig>(
        Configurable::BookEnabled,
        Option.getIntOption(USI_OPTION_BOOK_ENABLED)));
    Executor->pushCommand(std::make_shared<BoolConfig>(
        Configurable::RepetitionBookAllowed,
        Option.getBoolOption(USI_OPTION_REPETITION_BOOK_ALLOWED)));

    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::NumGPUs, Option.getIntOption(USI_OPTION_NUM_GPUS)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::NumSearchThreadsPerGPU,
        Option.getIntOption(USI_OPTION_NUM_SEARCH_THREADS)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::NumEvaluationThreadsPerGPU,
        Option.getIntOption(USI_OPTION_NUM_EVALUATION_THREADS_PER_GPU)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::NumCheckmateSearchThreads,
        Option.getIntOption(USI_OPTION_NUM_CHECKMATE_THREADS)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::BatchSize, Option.getIntOption(USI_OPTION_BATCH_SIZE)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::HashMemoryMB, Option.getIntOption(USI_OPTION_HASH)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::EvalCacheMemoryMB,
        Option.getIntOption(USI_OPTION_EVAL_CACHE_MEMORY_MB)));
    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::ThinkingTimeMargin,
        Option.getIntOption(USI_OPTION_THINKING_TIME_MARGIN)));

    Executor->pushCommand(std::make_shared<IntegerConfig>(
        Configurable::MaxPly, Option.getIntOption(USI_OPTION_MAX_PLY)));
    Executor->pushCommand(std::make_shared<DoubleConfig>(
        Configurable::BlackDrawValue,
        (double)Option.getIntOption(USI_OPTION_BLACK_DRAW_VALUE) / 100.0));
    Executor->pushCommand(std::make_shared<DoubleConfig>(
        Configurable::WhiteDrawValue,
        (double)Option.getIntOption(USI_OPTION_WHITE_DRAW_VALUE) / 100.0));

    Executor->pushCommand(std::make_shared<StringConfig>(
        Configurable::WeightPath,
        Option.getFileNameOption(USI_OPTION_WEIGHT_PATH)));
    Executor->pushCommand(std::make_shared<StringConfig>(
        Configurable::BookPath,
        Option.getFileNameOption(USI_OPTION_BOOK_PATH)));

    Executor->pushCommand(std::make_shared<GetReady>(), true);

    // Logger.setScoreFormatType(USILogger::ScoreFormatType::WinDraw);
    Logger->printRawMessage("readyok");
}

void position(std::istringstream& Stream) {
    std::string Token;
    std::string Sfen;

    Stream >> Token;

    if (Token == "startpos") {
        Sfen =
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 ";
    } else if (Token == "sfen") {
    } else {
        Logger->printLog("Unkwown token`", Token, "`.");
        return;
    }

    while (Stream >> Token) {
        Sfen += Token + " ";
    }

    Executor->pushCommand(
        std::make_shared<command::commands::SetPosition>(Sfen.c_str()));
    // State =
    // std::make_unique<nshogi::core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));
}

void bestMoveCallBackFunction(nshogi::core::Move32 Move, std::unique_ptr<mcts::ThoughtLog>) {
    Logger->printBestMove(Move);
}

void go(std::istringstream& Stream) {
    std::string Token;

    Limit Limits[2]{NoLimit, NoLimit};
    while (Stream >> Token) {
        if (Token == "btime") {
            Stream >> Limits[nshogi::core::Black].TimeLimitMilliSeconds;
        } else if (Token == "wtime") {
            Stream >> Limits[nshogi::core::White].TimeLimitMilliSeconds;
        } else if (Token == "binc") {
            Stream >> Limits[nshogi::core::Black].IncreaseMilliSeconds;
        } else if (Token == "winc") {
            Stream >> Limits[nshogi::core::White].IncreaseMilliSeconds;
        } else if (Token == "byoyomi") {
            uint32_t Byoyomi = 0;
            Stream >> Byoyomi;

            Limits[nshogi::core::Black].ByoyomiMilliSeconds = Byoyomi;
            Limits[nshogi::core::White].ByoyomiMilliSeconds = Byoyomi;
        } else if (Token == "infinite") {
            Limits[nshogi::core::Black] = NoLimit;
            Limits[nshogi::core::White] = NoLimit;
        } else {
            Logger->printLog("Unkwown token`", Token, "`.");
        }
    }

    Executor->pushCommand(std::make_shared<command::commands::Think>(
        Limits, bestMoveCallBackFunction));
}

void setOption(std::istringstream& Stream) {
    std::string Line;
    std::getline(Stream, Line);

    const std::size_t NamePos = Line.find(" name ");
    const std::size_t ValuePos = Line.find(" value ");

    if (NamePos == std::string::npos) {
        std::cout << "setoption error. `name` keyward was not found."
                  << std::endl;
        return;
    }

    if (ValuePos == std::string::npos) {
        std::cout << "setoption error. `value` keyward was not found."
                  << std::endl;
        return;
    }

    if (NamePos > ValuePos) {
        std::cout << "setoption error. `name` keyward must be followed by "
                     "`value` keyward."
                  << std::endl;
        return;
    }

    const std::string Name = Line.substr(NamePos + 6, ValuePos - 6);
    const std::string Value = Line.substr(ValuePos + 7);

    if (!Option.setOptionValue(Name.c_str(), Value.c_str())) {
        std::cout << "usioption of `" << Name << "` does not exist."
                  << std::endl;
    }
}

void stop() {
    Executor->pushCommand(std::make_shared<command::commands::Stop>());
}

void quit() {
    stop();

    Executor.reset(nullptr);
}

void debug() {
    const Context* C = Executor->getContext();
    std::cout << "===== ENGINE CONFIG =====" << std::endl;

    std::cout << "PonderingEnabled: " << C->getPonderingEnabled() << std::endl;

    std::cout << "=========================" << std::endl;
}

void nshogiExtension(std::istringstream& Stream) {
    std::string Token;
    Stream >> Token;

    if (Token == "makebook") {
        nshogi::engine::book::BookMaker bookMaker(Executor->getContext(), Logger);
        bookMaker.enumerateBookSeeds(10000);
    }
}

} // namespace

void mainLoop() {
    greet();

    std::string Line, Command;

    while (std::getline(std::cin, Line)) {
        std::istringstream Stream(Line);

        Stream >> std::skipws >> Command;

        if (Command == "isready") {
            isready();
        } else if (Command == "position") {
            position(Stream);
        } else if (Command == "go") {
            go(Stream);
        } else if (Command == "setoption") {
            setOption(Stream);
        } else if (Command == "stop") {
            stop();
        } else if (Command == "d" || Command == "debug") {
            debug();
        } else if (Command == "quit" || Command == "exit") {
            quit();
            break;
        } else if (Command == "nshogiext") {
            nshogiExtension(Stream);
        } else if (Command == "") {
            continue;
        } else {
            std::cout << "Unknown command `" << Command << "`." << std::endl;
        }

        Command = "";
    }
}

} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi
