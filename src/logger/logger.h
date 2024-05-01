#ifndef NSHOGI_ENGINE_LOGGER_LOGGER_H
#define NSHOGI_ENGINE_LOGGER_LOGGER_H


#include <cinttypes>
#include <vector>
#include <nshogi/core/types.h>


namespace nshogi {
namespace engine {
namespace logger {

struct PVLog {
    uint32_t ElapsedMilliSeconds = 0;
    double WinRate = 0;
    double DrawRate = 0;
    double DrawValue = 0.5;
    int32_t SolvedGameEndPly = 0;
    uint64_t NodesPerSecond = 0;
    uint64_t NumNodes = 0;
    uint64_t HashFullPerMille = 0;
    std::vector<core::Move16> PV;
};


class Logger {
 public:
    virtual void printPVLog(const PVLog& Log) const = 0;
    virtual void printBestMove(const core::Move32& Move) const = 0;
    virtual void printLog(const char* Message) const = 0;
};

} // namespace logger
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_LOGGER_H
