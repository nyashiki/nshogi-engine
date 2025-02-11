//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "framequeue.h"

namespace nshogi {
namespace engine {
namespace selfplay {

FrameQueue::FrameQueue()
    : IsClosed(false) {
}

void FrameQueue::add(std::unique_ptr<Frame>&& F) {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        if (!IsClosed) {
            Queue.push(std::move(F));
        }
    }
    CV.notify_one();
}

void FrameQueue::add(std::vector<std::unique_ptr<Frame>>& Fs) {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        if (!IsClosed) {
            for (auto&& F : Fs) {
                Queue.push(std::move(F));
            }
        }
    }
    CV.notify_one();
}


std::vector<std::unique_ptr<Frame>> FrameQueue::get(std::size_t Size, bool Wait, bool AcceptShortage) {
    std::vector<std::unique_ptr<Frame>> Buffer;

    {
        std::unique_lock<std::mutex> Lock(Mutex);

        if (Wait) {
            CV.wait(Lock, [this]() {
                return !Queue.empty() || IsClosed;
            });
        }

        if (AcceptShortage || Queue.size() >= Size) {
            while (!Queue.empty() && Buffer.size() < Size) {
                Buffer.emplace_back(std::move(Queue.front()));
                Queue.pop();
            }
        }
    }

    return Buffer;
}

std::queue<std::unique_ptr<Frame>> FrameQueue::getAll() {
    std::queue<std::unique_ptr<Frame>> Q;

    {
        std::unique_lock<std::mutex> Lock(Mutex);

        CV.wait(Lock, [this]() {
            return !Queue.empty() || IsClosed;
        });

        Queue.swap(Q);
    }

    return Q;
}

void FrameQueue::close() {
    {
        std::lock_guard<std::mutex> Lock(Mutex);
        IsClosed = true;
    }
    CV.notify_all();
}

} // namespace selfplay
} // namespace engine
} // namespace nshogi
