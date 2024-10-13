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
    FrameQueue();
    void add(std::unique_ptr<Frame>&&);
    void add(std::vector<std::unique_ptr<Frame>>&);
    auto get(std::size_t, bool Wait = true, bool AcceptShortage = true) -> std::vector<std::unique_ptr<Frame>>;
    auto getAll() -> std::queue<std::unique_ptr<Frame>>;
    void close();

 private:
    std::queue<std::unique_ptr<Frame>> Queue;
    bool IsClosed;
    std::mutex Mutex;
    std::condition_variable CV;
};

} // namespace selfplay
} // namespace engine
} // namespcae nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_FRAMEQUEUE_H
