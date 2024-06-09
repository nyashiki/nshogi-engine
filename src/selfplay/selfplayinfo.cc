#include "selfplayinfo.h"

#include <cassert>

namespace nshogi {
namespace engine {
namespace selfplay {

SelfplayInfo::SelfplayInfo(std::size_t OnGoingGames)
    : NumOnGoingGames(OnGoingGames) {
}

void SelfplayInfo::decrementNumOnGoingGames() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        assert(NumOnGoingGames > 0);
        --NumOnGoingGames;
    }
    CV.notify_all();
}

void SelfplayInfo::waitUntilAllGamesFinished() {
    std::unique_lock<std::mutex> Lock(Mutex);
    CV.wait(Lock, [this]() { return NumOnGoingGames == 0; });
}

std::size_t SelfplayInfo::getNumOnGoinggames() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return NumOnGoingGames;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
