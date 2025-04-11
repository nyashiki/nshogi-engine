//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "nshogi.h"
#include "onnxruntime.h"

#include <inttypes.h>

#ifndef MCTS_H
#define MCTS_H

typedef struct node {
    struct node* parent;
    nshogi_move_t move;
    nshogi_repetition_t repetition;

    int num_children;
    struct node** children;

    float policy_from_parent;

    float predicted_win_rate;
    float predicted_draw_rate;
    double accumulated_win_rate;
    double accumulated_draw_rate;

    uint64_t visit_count;
} node_t;

typedef struct tree {
    node_t* root;
} tree_t;

nshogi_move_t startSearch(nshogi_state_t* state, nshogi_state_config_t* state_config, int num_simulation, onnxruntime_t* onnx_runtime);

#endif // #ifndef MCTS_H
