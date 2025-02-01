//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <vector>

#include <nshogi/core/positionbuilder.h>
#include <nshogi/core/types.h>

namespace nshogi {
namespace engine {
namespace selfplay {

class PositionBuilderShogi816k : public core::PositionBuilder {
 public:
    PositionBuilderShogi816k();
    core::Position build();

 private:
    void shuffle();

    core::Square Sliders[2][9];
    core::Square Steps[2][9];
};

} // namespace selfplay
} // namespace engine
} // namespace nshogi
