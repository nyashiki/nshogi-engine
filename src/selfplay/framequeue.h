#ifndef NSHOGI_ENGINE_SELFPLAY_FRAMEQUEUE_H
#define NSHOGI_ENGINE_SELFPLAY_FRAMEQUEUE_H

#include "frame.h"

#include <condition_variable>
#include <memory>
#include <queue>

namespace nshogi {
namespace engine {
namespace selfplay {

class FrameQueue {
 public:
    void add(std::unique_ptr<Frame>&&);
    auto get(std::size_t) -> std::vector<std::unique_ptr<Frame>>;
    auto getAll() -> std::queue<std::unique_ptr<Frame>>;

 private:
    std::queue<std::unique_ptr<Frame>> Queue;
    std::mutex Mutex;
    std::condition_variable CV;
};

} // namespace selfplay
} // namespace engine
} // namespcae nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_FRAMEQUEUE_H
