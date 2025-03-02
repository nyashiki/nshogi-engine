//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#ifndef NSHOGI_ENGINE_INFER_MIGRAPHX_H
#define NSHOGI_ENGINE_INFER_MIGRAPHX_H

#include "infer.h"

#include <string>

namespace nshogi {
namespace engine {
namespace infer {

class MIGraphX : public Infer {
 public:
    MIGraphX();
    ~MIGraphX() override;

    void load(const std::string& Path);

    void computeNonBlocking(const ml::FeatureBitboard*, std::size_t BatchSize,
                            float* DstPolicy, float* DstWinRate,
                            float* DstDrawRate) override;
    void computeBlocking(const ml::FeatureBitboard*, std::size_t BatchSize,
                         float* DstPolicy, float* DstWinRate,
                         float* DstDrawRate) override;
    void await() override;
    bool isComputing() override;
};

} // namespace infer
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_INFER_MIGRAPHX_H
