//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_SELFPLAY_PHASE_H
#define NSHOGI_ENGINE_SELFPLAY_PHASE_H

#include <cinttypes>

namespace nshogi {
namespace engine {
namespace selfplay {

enum class SelfplayPhase : uint8_t {
    Initialization,
    RootPreparation,
    LeafSelection,
    LeafTerminalChecking,
    Evaluation,
    Backpropagation,
    SequentialHalving,
    Judging,
    Transition,
    Save,
};

} // namespace selfpaly
} // namespace engine
} // nshogi

#endif // #ifndef NSHOGI_ENGINE_SELFPLAY_PHASE_H
