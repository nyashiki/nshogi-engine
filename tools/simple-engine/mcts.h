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

///
/// @struct node_t
/// @brief MCTS node.
///
typedef struct node {
    /// @brief The parent node of this node.
    struct node* parent;

    /// @brief The move from the parent to this node.
    nshogi_move_t move;

    /// @brief Repetition status of the state of this node.
    nshogi_repetition_t repetition;

    /// @brief The number of children that this node has.
    int num_children;

    /// @brief The children that this node has.
    struct node** children;

    /// @brief The policy probabiltiy from the parent to this node.
    float policy_from_parent;

    /// @brief The win rate predicted by neural networks.
    float predicted_win_rate;

    /// @brief The draw rate predicted by neural networks.
    float predicted_draw_rate;

    /// @brief The sum of win rate of the sub-tree from this node.
    double accumulated_win_rate;

    /// @brief The sum of draw rate of the sub-tree from this node.
    double accumulated_draw_rate;

    uint64_t visit_count;
} node_t;

///
/// @struct tree_t
/// @brief Search tree.
///
typedef struct tree {
    node_t* root;
} tree_t;

///
/// @fn startSearch
/// @brief Start search.
/// @param state The state at which the search starts.
/// @param state_config The state config.
/// @param num_simulation The number of simulations that MCTS conducts.
/// @param onnx_runtime The instance of onnxruntime_t.
/// @param temperature Sampling temperature of selecting next move after search finishes.
/// @return A next move.
///
nshogi_move_t startSearch(
        nshogi_state_t* state,
        nshogi_state_config_t* state_config,
        int num_simulation,
        onnxruntime_t* onnx_runtime,
        float temperature);

#endif // #ifndef MCTS_H
