#include "usi.h"

#include "usioption.h"
#include "../globalconfig.h"
#include "../limit.h"
#include "../evaluate/batch.h"
#include "../evaluate/evaluator.h"
#include "../evaluate/preset.h"
#include "../infer/random.h"
#include "../mcts/searchworker.h"
#include "../mcts/manager.h"
#include "usilogger.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <memory>
#include <sstream>
#include <cstdio>

#include <nshogi/book/book.h>
#include <nshogi/core/state.h>
#include <nshogi/core/stateconfig.h>
#include <nshogi/core/movegenerator.h>
#include <nshogi/io/sfen.h>

#include <nshogi/ml/featurestack.h>
#include <thread>

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

namespace {

std::unique_ptr<mcts::Manager> Manager;
std::unique_ptr<nshogi::core::State> State;
std::unique_ptr<nshogi::core::StateConfig> StateConfig =
    std::make_unique<nshogi::core::StateConfig>();
std::unique_ptr<nshogi::book::Book> Book = nullptr;

USIOption Option;
std::shared_ptr<USILogger> Logger = std::make_shared<USILogger>();

Limit Limits[nshogi::core::NumColors];

} // namespace

namespace {

void setupOption() {
    Option.addIntOption("USI_MaxPly", 320, 1, 99999);
    Option.addIntOption("USI_Hash",
            (int64_t)GlobalConfig::getConfig().getAvailableMemoryMB(), 1024LL, 1024 * 1024LL);
    Option.addBoolOption("USI_Ponder",
            GlobalConfig::getConfig().getPonderingEnabled());
    Option.addIntOption("NumGPUs",
            (int64_t)GlobalConfig::getConfig().getNumGPUs(), 1, 16);
    Option.addIntOption("NumSearchThreads",
            (int64_t)GlobalConfig::getConfig().getNumSearchThreads(), 1, 2048);
    Option.addIntOption("NumEvaluationThreadsPerGPU",
            (int64_t)GlobalConfig::getConfig().getNumEvaluationThreadsPerGPU(), 1, 2048);
    Option.addIntOption("NumCheckmateSearchThreads",
            (int64_t)GlobalConfig::getConfig().getNumCheckmateSearchThreads(), 0, 128);
    Option.addIntOption("BatchSize",
            (int64_t)GlobalConfig::getConfig().getBatchSize(), 1, 4096);
    Option.addBoolOption("IsBookEnabled",
            GlobalConfig::getConfig().isBookEnabled());
    Option.addFileNameOption("WeightPath",
            GlobalConfig::getConfig().getWeightPath().c_str());
    Option.addFileNameOption("BookPath",
            GlobalConfig::getConfig().getBookPath().c_str());
    Option.addIntOption("EvalCacheMemoryMB",
            (int64_t)GlobalConfig::getConfig().getEvalCacheMemoryMB(), 0LL, 1024 * 1024LL);
    Option.addIntOption("ThinkingTimeMargin",
            (int64_t)GlobalConfig::getConfig().getThinkingTimeMargin(), 0LL, 60 * 1000);
    Option.addIntOption("BlackDrawValue",
            (int)(GlobalConfig::getConfig().getBlackDrawValue() * 100.0f), 0, 100);
    Option.addIntOption("WhiteDrawValue",
            (int)(GlobalConfig::getConfig().getWhiteDrawValue() * 100.0f), 0, 100);
    Option.addBoolOption("RepetitionBookAllowed",
            GlobalConfig::getConfig().isRepetitionBookAllowed());
}

void showOption() {
    Option.showOption();
}

void greet() {
    setupOption();

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

    GlobalConfig::getConfig().setNumGPUs(
            (std::size_t)(Option.getIntOption("NumGPUs")));
    GlobalConfig::getConfig().setNumSearchThreads(
            (std::size_t)(Option.getIntOption("NumSearchThreads")));
    GlobalConfig::getConfig().setNumEvaluationThreadsPerGPU(
            (std::size_t)(Option.getIntOption("NumEvaluationThreadsPerGPU")));
    GlobalConfig::getConfig().setNumCheckmateSearchThreads(
            (std::size_t)(Option.getIntOption("NumCheckmateSearchThreads")));
    GlobalConfig::getConfig().setBatchSize(
            (std::size_t)(Option.getIntOption("BatchSize")));
    GlobalConfig::getConfig().setAvailableMemoryMB(
            (std::size_t)(Option.getIntOption("USI_Hash")));
    GlobalConfig::getConfig().setPonderingEnabled(
            Option.getBoolOption("USI_Ponder"));
    GlobalConfig::getConfig().setWeightPath(
            std::string(Option.getFileNameOption("WeightPath")));
    GlobalConfig::getConfig().setIsBookEnabled(
            Option.getBoolOption("IsBookEnabled"));
    GlobalConfig::getConfig().setBookPath(
            std::string(Option.getFileNameOption("BookPath")));
    GlobalConfig::getConfig().setEvalCacheMemoryMB(
            (std::size_t)(Option.getIntOption("EvalCacheMemoryMB")));
    GlobalConfig::getConfig().setThinkingTimeMargin(
            (uint32_t)(Option.getIntOption("ThinkingTimeMargin")));
    GlobalConfig::getConfig().setBlackDrawValue(
            (float)Option.getIntOption("BlackDrawValue") / 100.0f);
    GlobalConfig::getConfig().setWhiteDrawValue(
            (float)Option.getIntOption("WhiteDrawValue") / 100.0f);
    GlobalConfig::getConfig().setIsRepetitionBookAllowed(
            Option.getBoolOption("RepetitionBookAllowed"));

    const std::size_t AvailableMemory = GlobalConfig::getConfig().getAvailableMemoryMB() * 1024ULL * 1024ULL;
    nshogi::engine::allocator::getNodeAllocator().resize((std::size_t)(0.1 * (double)AvailableMemory));
    nshogi::engine::allocator::getEdgeAllocator().resize((std::size_t)(0.9 * (double)AvailableMemory));

    if (GlobalConfig::getConfig().isBookEnabled()) {
        // Book = std::make_unique<nshogi::book::Book>(nshogi::book::Book::loadYaneuraOuFormat(
        //             GlobalConfig::getConfig().getBookPath().c_str()));
        Book = std::make_unique<nshogi::book::Book>(nshogi::book::Book::load(
                    GlobalConfig::getConfig().getBookPath().c_str()));

        Logger->printLog("Book->size(): ", Book->size());
    }

    Manager = std::make_unique<mcts::Manager>(
            GlobalConfig::getConfig().getBatchSize(),
            GlobalConfig::getConfig().getNumGPUs(),
            GlobalConfig::getConfig().getNumSearchThreads(),
            GlobalConfig::getConfig().getNumEvaluationThreadsPerGPU(),
            GlobalConfig::getConfig().getNumCheckmateSearchThreads(),
            GlobalConfig::getConfig().getEvalCacheMemoryMB(),
            Logger);

    Manager->setIsPonderingEnabled(GlobalConfig::getConfig().getPonderingEnabled());

    StateConfig->Rule = core::Declare27_ER;
    StateConfig->MaxPly = (uint16_t)Option.getIntOption("USI_MaxPly");
    StateConfig->BlackDrawValue = GlobalConfig::getConfig().getBlackDrawValue();
    StateConfig->WhiteDrawValue = GlobalConfig::getConfig().getWhiteDrawValue();

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

    State = std::make_unique<nshogi::core::State>(nshogi::io::sfen::StateBuilder::newState(Sfen));
}

void go(std::istringstream& Stream, void (*CallBack)(nshogi::core::Move32 Move)) {
    assert(Manager != nullptr);

    std::string Token;

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

    if (Book != nullptr) {
        if (State->getRepetitionStatus() == nshogi::core::RepetitionStatus::NoRepetition ||
                (State->getRepetitionStatus() == nshogi::core::RepetitionStatus::Repetition &&
                 GlobalConfig::getConfig().isRepetitionBookAllowed())) {
            nshogi::book::Entry* Entry =
                Book->findEntry(nshogi::io::sfen::positionToSfen(State->getPosition()).c_str());

            if (Entry != nullptr) {
                Logger->printLog("info string Entry->getNumBookMoves(): ", (int)Entry->getNumBookMoves());

                const uint8_t NumBookMoves = Entry->getNumBookMoves();
                static std::random_device SeedGen;
                static std::mt19937_64 Mt(SeedGen());

                if (NumBookMoves > 0) {
                    if (GlobalConfig::getConfig().getBookSelectionStrategy() == book::Strategy::MostVisited) {
                        uint64_t MaxCount = 0;
                        nshogi::book::BookMove* MaxCountBookMove = nullptr;

                        for (std::size_t I = 0; I < Entry->getNumBookMoves(); ++I) {
                            nshogi::book::BookMove* BookMove = Entry->getBookMove(I);

                            Logger->printLog(nshogi::io::sfen::move32ToSfen(BookMove->getMove()),
                                            ", ",
                                            BookMove->getMeta().getCount());

                            if (BookMove->getMeta().getCount() > MaxCount) {
                                MaxCount = BookMove->getMeta().getCount();
                                MaxCountBookMove = BookMove;
                            }
                        }

                        auto Move = MaxCountBookMove->getMove();
                        CallBack(Move);
                    } else if (GlobalConfig::getConfig().getBookSelectionStrategy() == book::Strategy::Random) {
                        nshogi::book::BookMove* BookMove = Entry->getBookMove((uint8_t)Mt() % NumBookMoves);
                        auto Move = BookMove->getMove();
                        CallBack(Move);
                    }

                    return;
                }
            }
        }
    }

    Manager->thinkNextMove(*State, *StateConfig, Limits[State->getSideToMove()], CallBack);
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
    Manager->interrupt();
}

void quit() {
    stop();
    Manager.reset();
}

void debug() {
    std::cout << "Sfen: " << nshogi::io::sfen::stateToSfen(*State) << std::endl;
    printf("Hash: 0x%lx\n", State->getHash());

    const auto LegalMoves = nshogi::core::MoveGenerator::generateLegalMoves(*State);
    std::cout << "Legal moves:";
    for (const auto& Move : LegalMoves) {
        std::cout << " " << nshogi::io::sfen::move32ToSfen(Move);
    }
    std::cout << std::endl;

    Option.showOption();
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
