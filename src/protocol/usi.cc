#include "usi.h"

#include "../command/executor.h"
#include "usioption.h"
#include "usilogger.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <memory>
#include <sstream>
#include <cstdio>

#include <nshogi/book/book.h>
#include <nshogi/core/state.h>
#include <nshogi/core/types.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

#include <thread>

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

void setupOption(const Context* C) {
    Option.addIntOption("USI_MaxPly", 320, 1, 99999);
    Option.addIntOption("USI_Hash",
            (int64_t)C->getAvailableMemoryMB(), 1024LL, 1024 * 1024LL);
    Option.addBoolOption("USI_Ponder",
            C->getPonderingEnabled());
    Option.addIntOption("NumGPUs",
            (int64_t)C->getNumGPUs(), 1, 16);
    Option.addIntOption("NumSearchThreads",
            (int64_t)C->getNumSearchThreads(), 1, 2048);
    Option.addIntOption("NumEvaluationThreadsPerGPU",
            (int64_t)C->getNumEvaluationThreadsPerGPU(), 1, 2048);
    Option.addIntOption("NumCheckmateSearchThreads",
            (int64_t)C->getNumCheckmateSearchThreads(), 0, 128);
    Option.addIntOption("BatchSize",
            (int64_t)C->getBatchSize(), 1, 4096);
    Option.addBoolOption("IsBookEnabled",
            C->isBookEnabled());
    Option.addFileNameOption("WeightPath",
            C->getWeightPath().c_str());
    Option.addFileNameOption("BookPath",
            C->getBookPath().c_str());
    Option.addIntOption("EvalCacheMemoryMB",
            (int64_t)C->getEvalCacheMemoryMB(), 0LL, 1024 * 1024LL);
    Option.addIntOption("ThinkingTimeMargin",
            (int64_t)C->getThinkingTimeMargin(), 0LL, 60 * 1000);
    Option.addIntOption("BlackDrawValue",
            (int)(C->getBlackDrawValue() * 100.0f), 0, 100);
    Option.addIntOption("WhiteDrawValue",
            (int)(C->getWhiteDrawValue() * 100.0f), 0, 100);
    Option.addBoolOption("RepetitionBookAllowed",
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

    Executor->pushCommand(std::make_unique<BoolConfig>(
                Configurable::PonderEnabled,
                Option.getIntOption("USI_Ponder")));
    Executor->pushCommand(std::make_unique<BoolConfig>(
                Configurable::BookEnabled,
                Option.getIntOption("IsBookEnabled")));
    Executor->pushCommand(std::make_unique<BoolConfig>(
                Configurable::RepetitionBookAllowed,
                Option.getBoolOption("RepetitionBookAllowed")));

    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::NumGPUs, Option.getIntOption("NumGPUs")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::NumSearchThreadsPerGPU,
                Option.getIntOption("NumSearchThreads")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::NumEvaluationThreadsPerGPU,
                Option.getIntOption("NumEvaluationThreadsPerGPU")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::NumCheckmateSearchThreads,
                Option.getIntOption("NumCheckmateSearchThreads")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::BatchSize,
                Option.getIntOption("BatchSize")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::HashMemoryMB,
                Option.getIntOption("USI_Hash")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::EvalCacheMemoryMB,
                Option.getIntOption("EvalCacheMemoryMB")));
    Executor->pushCommand(std::make_unique<IntegerConfig>(
                Configurable::ThinkingTimeMargin,
                Option.getIntOption("ThinkingTimeMargin")));

    Executor->pushCommand(std::make_unique<DoubleConfig>(
                Configurable::BlackDrawValue,
                (double)Option.getIntOption("BlackDrawValue") / 100.0));
    Executor->pushCommand(std::make_unique<DoubleConfig>(
                Configurable::WhiteDrawValue,
                (double)Option.getIntOption("WhiteDrawValue") / 100.0));

    Executor->pushCommand(std::make_unique<StringConfig>(
                Configurable::WeightPath,
                Option.getFileNameOption("WeightPath")));
    Executor->pushCommand(std::make_unique<StringConfig>(
                Configurable::BookPath,
                Option.getFileNameOption("BookPath")));

    Executor->pushCommand(std::make_unique<GetReady>());

    // const std::size_t AvailableMemory = GlobalConfig::getConfig().getAvailableMemoryMB() * 1024ULL * 1024ULL;
    // nshogi::engine::allocator::getNodeAllocator().resize((std::size_t)(0.1 * (double)AvailableMemory));
    // nshogi::engine::allocator::getEdgeAllocator().resize((std::size_t)(0.9 * (double)AvailableMemory));

    // StateConfig->Rule = core::ER_Declare27;
    // StateConfig->MaxPly = (uint16_t)Option.getIntOption("USI_MaxPly");
    // StateConfig->BlackDrawValue = GlobalConfig::getConfig().getBlackDrawValue();
    // StateConfig->WhiteDrawValue = GlobalConfig::getConfig().getWhiteDrawValue();

    // Logger.setScoreFormatType(USILogger::ScoreFormatType::WinDraw);
    Logger->printRawMessage("readyok");
}

void position(std::istringstream& Stream) {
    std::string Token;
    std::string Sfen;

    Stream >> Token;

    if (Token == "startpos") {
        Sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 ";
    } else if (Token == "sfen") {
    } else {
        Logger->printLog("Unkwown token`", Token, "`.");
        return;
    }

    while (Stream >> Token) {
        Sfen += Token + " ";
    }

    // State = std::make_unique<nshogi::core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));
}

void go(std::istringstream& Stream, void (*CallBack)(nshogi::core::Move32 Move)) {
    std::string Token;

    // while (Stream >> Token) {
    //     if (Token == "btime") {
    //         Stream >> Limits[nshogi::core::Black].TimeLimitMilliSeconds;
    //     } else if (Token == "wtime") {
    //         Stream >> Limits[nshogi::core::White].TimeLimitMilliSeconds;
    //     } else if (Token == "binc") {
    //         Stream >> Limits[nshogi::core::Black].IncreaseMilliSeconds;
    //     } else if (Token == "winc") {
    //         Stream >> Limits[nshogi::core::White].IncreaseMilliSeconds;
    //     } else if (Token == "byoyomi") {
    //         uint32_t Byoyomi = 0;
    //         Stream >> Byoyomi;

    //         Limits[nshogi::core::Black].ByoyomiMilliSeconds = Byoyomi;
    //         Limits[nshogi::core::White].ByoyomiMilliSeconds = Byoyomi;
    //     } else if (Token == "infinite") {
    //         Limits[nshogi::core::Black] = NoLimit;
    //         Limits[nshogi::core::White] = NoLimit;
    //     } else {
    //         Logger->printLog("Unkwown token`", Token, "`.");
    //     }
    // }
}

void setOption(std::istringstream& Stream) {
    std::string Line;
    std::getline(Stream, Line);

    const std::size_t NamePos = Line.find(" name ");
    const std::size_t ValuePos = Line.find(" value ");

    if (NamePos == std::string::npos) {
        std::cout << "setoption error. `name` keyward was not found." << std::endl;
        return;
    }

    if (ValuePos == std::string::npos) {
        std::cout << "setoption error. `value` keyward was not found." << std::endl;
        return;
    }

    if (NamePos > ValuePos) {
        std::cout << "setoption error. `name` keyward must be followed by `value` keyward." << std::endl;
        return;
    }

    const std::string Name = Line.substr(NamePos + 6, ValuePos - 6);
    const std::string Value = Line.substr(ValuePos + 7);

    if (!Option.setOptionValue(Name.c_str(), Value.c_str())) {
        std::cout << "usioption of `" << Name << "` does not exist." << std::endl;
    }
}

void stop() {
}

void quit() {
    stop();

    Executor.reset(nullptr);
}

void debug() {
}

void bestMoveCallBackFunction(nshogi::core::Move32 Move) {
    Logger->printBestMove(Move);
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
            go(Stream, bestMoveCallBackFunction);
        } else if (Command == "setoption") {
            setOption(Stream);
        } else if (Command == "stop") {
            stop();
        } else if (Command == "d" || Command == "debug") {
            debug();
        } else if (Command == "quit" || Command == "exit") {
            quit();
            break;
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
