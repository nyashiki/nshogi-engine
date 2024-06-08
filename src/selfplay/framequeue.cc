#include "framequeue.h"

namespace nshogi {
namespace engine {
namespace selfplay {

void FrameQueue::add(std::unique_ptr<Frame>&& F) {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        Queue.push(std::move(F));
    }
    CV.notify_one();
}

std::vector<std::unique_ptr<Frame>> FrameQueue::get(std::size_t Size) {
    std::vector<std::unique_ptr<Frame>> Buffer;

    while (Buffer.size() < Size) {
        std::unique_ptr<Frame> F;
        bool IsQueueEmpty = false;
        {
            std::unique_lock<std::mutex> Lock(Mutex);

            CV.wait(Lock, [this]() {
                return !Queue.empty();
            });

            F = std::move(Queue.front());
            Queue.pop();

            IsQueueEmpty = Queue.empty();
        }

        Buffer.emplace_back(std::move(F));
        if (IsQueueEmpty) {
            break;
        }
    }

    return Buffer;
}

std::queue<std::unique_ptr<Frame>> FrameQueue::getAll() {
    std::queue<std::unique_ptr<Frame>> Q;

    {
        std::unique_lock<std::mutex> Lock(Mutex);

        CV.wait(Lock, [this]() {
            return !Queue.empty();
        });

        Queue.swap(Q);
    }

    return Q;
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
