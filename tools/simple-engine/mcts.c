//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "mcts.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

///
/// @fn getRandom
/// @brief Sampling a random number in [0, 1).
/// @return A random number in [0, 1).
///
static double getRandom() {
    return rand() / ((double)RAND_MAX);
}

///
/// @fn softmax
/// @brief Do inplace softmax computation.
/// @param x Input.
/// @param num The number of elements `x` has.
/// @param temperature Softmax temperature.
///
static void softmax(float* x, int num, float temperature) {
    // For stability of softmax computation,
    // you can use \propto e^{x_i - M} instead of \propto e^{x_i},
    // where M is an arbitrary number, which is M = \max x_i in general.

    // Get the max.
    float max_value = x[0];
    for (int i = 1; i < num; ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0;

    // Compute e^{x_i - M}.
    for (int i = 0; i < num; ++i) {
        x[i] = exp((x[i] - max_value) / temperature);
        sum += x[i];
    }

    // Normalization: dividing by the sum of e^{x_i - M}.
    for (int i = 0; i < num; ++i) {
        x[i] /= sum;
    }
}

///
/// @fn mcts_policy_with_temperature
/// @brief Do inplace normalization computation.
/// @param x Input.
/// @param num The number of elements `x` has.
/// @param temperature temperature.
///
static void mcts_policy_with_temperature(float* x, int num, float temperature) {
    float sum = 0.0f;

    // Compute x_i^{1/temperature}.
    for (int i = 0; i < num; ++i) {
        x[i] = powf(x[i], 1.0f / temperature);
        sum += x[i];
    }

    // Normalization: dividing by the sum of x_i^{1/temperature}.
    for (int i = 0; i < num; ++i) {
        x[i] /= sum;
    }
}

///
/// @fn winRateToScore
/// @brief Convert win rate to centi-pawn score.
///
static int winRateToScore(double x) {
    const double ponanzaConstant = 600.0;

    if (x <= 0.0) {
        return -9999;
    } else if (x >= 1.0) {
        return 9999;
    }

    return (int)(-ponanzaConstant * log(1.0 / x - 1.0));
}

///
/// @fn makeFeature
/// @break Make feature vector by state and state config.
/// @param dest The destination on which the feature vectore is stored.
///
static void makeFeature(float* dest, nshogi_state_t* state, nshogi_state_config_t* state_config) {
    nshogi_feature_type_t features[] = {
        NSHOGI_FT_MYPAWN, NSHOGI_FT_MYLANCE,
        NSHOGI_FT_MYKNIGHT, NSHOGI_FT_MYSILVER,
        NSHOGI_FT_MYGOLD, NSHOGI_FT_MYKING,
        NSHOGI_FT_MYBISHOP, NSHOGI_FT_MYROOK,
        NSHOGI_FT_MYPROPAWN, NSHOGI_FT_MYPROLANCE,
        NSHOGI_FT_MYPROKNIGHT, NSHOGI_FT_MYPROSILVER,
        NSHOGI_FT_MYPROBISHOP, NSHOGI_FT_MYPROROOK,
        NSHOGI_FT_OPPAWN, NSHOGI_FT_OPLANCE,
        NSHOGI_FT_OPKNIGHT, NSHOGI_FT_OPSILVER,
        NSHOGI_FT_OPGOLD, NSHOGI_FT_OPKING,
        NSHOGI_FT_OPBISHOP, NSHOGI_FT_OPROOK,
        NSHOGI_FT_OPPROPAWN, NSHOGI_FT_OPPROLANCE,
        NSHOGI_FT_OPPROKNIGHT, NSHOGI_FT_OPPROSILVER,
        NSHOGI_FT_OPPROBISHOP, NSHOGI_FT_OPPROROOK,
        NSHOGI_FT_MYSTANDPAWN1, NSHOGI_FT_MYSTANDPAWN2,
        NSHOGI_FT_MYSTANDPAWN3, NSHOGI_FT_MYSTANDPAWN4,
        NSHOGI_FT_MYSTANDPAWN5, NSHOGI_FT_MYSTANDPAWN6,
        NSHOGI_FT_MYSTANDLANCE1, NSHOGI_FT_MYSTANDLANCE2,
        NSHOGI_FT_MYSTANDLANCE3, NSHOGI_FT_MYSTANDLANCE4,
        NSHOGI_FT_MYSTANDKNIGHT1, NSHOGI_FT_MYSTANDKNIGHT2,
        NSHOGI_FT_MYSTANDKNIGHT3, NSHOGI_FT_MYSTANDKNIGHT4,
        NSHOGI_FT_MYSTANDSILVER1, NSHOGI_FT_MYSTANDSILVER2,
        NSHOGI_FT_MYSTANDSILVER3, NSHOGI_FT_MYSTANDSILVER4,
        NSHOGI_FT_MYSTANDGOLD1, NSHOGI_FT_MYSTANDGOLD2,
        NSHOGI_FT_MYSTANDGOLD3, NSHOGI_FT_MYSTANDGOLD4,
        NSHOGI_FT_MYSTANDBISHOP1, NSHOGI_FT_MYSTANDBISHOP2,
        NSHOGI_FT_MYSTANDROOK1, NSHOGI_FT_MYSTANDROOK2,
        NSHOGI_FT_OPSTANDPAWN1, NSHOGI_FT_OPSTANDPAWN2,
        NSHOGI_FT_OPSTANDPAWN3, NSHOGI_FT_OPSTANDPAWN4,
        NSHOGI_FT_OPSTANDPAWN5, NSHOGI_FT_OPSTANDPAWN6,
        NSHOGI_FT_OPSTANDLANCE1, NSHOGI_FT_OPSTANDLANCE2,
        NSHOGI_FT_OPSTANDLANCE3, NSHOGI_FT_OPSTANDLANCE4,
        NSHOGI_FT_OPSTANDKNIGHT1, NSHOGI_FT_OPSTANDKNIGHT2,
        NSHOGI_FT_OPSTANDKNIGHT3, NSHOGI_FT_OPSTANDKNIGHT4,
        NSHOGI_FT_OPSTANDSILVER1, NSHOGI_FT_OPSTANDSILVER2,
        NSHOGI_FT_OPSTANDSILVER3, NSHOGI_FT_OPSTANDSILVER4,
        NSHOGI_FT_OPSTANDGOLD1, NSHOGI_FT_OPSTANDGOLD2,
        NSHOGI_FT_OPSTANDGOLD3, NSHOGI_FT_OPSTANDGOLD4,
        NSHOGI_FT_OPSTANDBISHOP1, NSHOGI_FT_OPSTANDBISHOP2,
        NSHOGI_FT_OPSTANDROOK1, NSHOGI_FT_OPSTANDROOK2,
        NSHOGI_FT_BLACK, NSHOGI_FT_WHITE,
        NSHOGI_FT_CHECK, NSHOGI_FT_NOMYPAWNFILE,
        NSHOGI_FT_NOOPPAWNFILE, NSHOGI_FT_PROGRESS,
        NSHOGI_FT_PROGRESSUNIT,
        NSHOGI_FT_MYDRAWVALUE, NSHOGI_FT_OPDRAWVALUE,
        NSHOGI_FT_MYDECLARATIONSCORE,
        NSHOGI_FT_OPDECLARATIONSCORE, NSHOGI_FT_MYPIECESCORE,
        NSHOGI_FT_OPPIECESCORE
    };

    int num_features = sizeof(features) / sizeof(features[0]);
    ml_api->makeFeatureVector(dest, state, state_config, features, num_features);
}

///
/// @fn initializeNode
/// @brief Initialize a node.
/// @param node Node to initialize.
///
static void initializeNode(node_t* node) {
    node->parent = NULL;
    node->move = 0;
    node->repetition = NSHOGI_NO_REPETITION;

    node->num_children = 0;
    node->children = NULL;

    node->policy_from_parent = 0;

    node->predicted_win_rate = 0;
    node->predicted_draw_rate = 0;
    node->accumulated_win_rate = 0;
    node->accumulated_draw_rate = 0;

    node->visit_count = 0;
}

///
/// @fn destroyNode
/// @brief Destroy a node.
/// @param node Node to destroy.
///
static void destroyNode(node_t* node) {
    // If the node has children, recursively destroy them.
    if (node->num_children != 0) {
        for (int i = 0; i < node->num_children; ++i) {
            destroyNode(node->children[i]);
        }
        free(node->children);
    }

    free(node);
}

///
/// @fn expandNode
/// @brief Expand children.
///
static void expandNode(nshogi_state_t* state, node_t* node) {
    nshogi_move_t moves[600];
    node->num_children = state_api->generateMoves(state, 1, moves);

    if (node->num_children > 0) {
        node->children = (node_t**)malloc(node->num_children * sizeof(node_t*));
    }

    for (int i = 0; i < node->num_children; ++i) {
        node->children[i] = (node_t*)malloc(sizeof(node_t));

        initializeNode(node->children[i]);

        node->children[i]->parent = node;
        node->children[i]->move = moves[i];
    }
}

///
/// @fn selectChildToSearch
/// @brief Select the best child to search according to pUCT formula.
///
static node_t* selectChildToSearch(node_t* node, nshogi_state_t* state, nshogi_state_config_t* state_config) {
    node_t* ucb_best_child = NULL;
    double ucb_best_value = 0;

    double c_base = 19652;
    double c_init = 1.25;
    double c = log((double)((1 + node->visit_count + c_base) / c_base)) + c_init;
    double n_sqrt = sqrt((double)node->visit_count);
    double c_n_sqrt = c * n_sqrt;

    float draw_value = state_api->getSideToMove(state) == NSHOGI_BLACK
        ? state_api->getBlackDrawValue(state_config)
        : state_api->getWhiteDrawValue(state_config);

    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];

        double q = 0;
        if (child->visit_count > 0) {
            double child_win_rate = child->accumulated_win_rate / child->visit_count;
            double child_draw_rate = child->accumulated_draw_rate / child->visit_count;
            q = (1.0 - child_draw_rate) * (1.0 - child_win_rate) + child_draw_rate * draw_value;
        }

        double u = c_n_sqrt * child->policy_from_parent / (1 + child->visit_count);
        double ucb = q + u;

        if (ucb_best_child == NULL || ucb > ucb_best_value) {
            ucb_best_value = ucb;
            ucb_best_child = child;
        }
    }

    return ucb_best_child;
}

///
/// @fn pickUpLeafNode
/// @brief Select the best leaf node to expand and evaluate.
///
static node_t* pickUpLeafNode(tree_t* tree, nshogi_state_t* state, nshogi_state_config_t* state_config) {
    node_t* node = tree->root;

    while (1) {
        // Reached the leaf node.
        if (node->visit_count == 0) {
            break;
        }

        // Reached a terminal node.
        if (node->num_children == 0) {
            break;
        }

        // Reached a repeated state.
        // When the root node is repeated, exclude it for search to proceed.
        if (node != tree->root && node->repetition != NSHOGI_NO_REPETITION) {
            break;
        }

        // Recursively select the best child node to search
        // until the node is a leaf node.
        node = selectChildToSearch(node, state, state_config);
        state_api->doMove(state, node->move);
    }

    return node;
}

///
/// @fn evaluateNode
/// @brief Evaluate a (leaf) node.
///
static void evaluateNode(nshogi_state_t* state, nshogi_state_config_t* state_config, node_t* node, onnxruntime_t* onnx_runtime) {
    // Checkmate.
    if (node->num_children == 0) {
        if (nshogi_api->isDroppingPawn(node->move)) {
            // Checkmate by a dropping pawn, which is prohibited by the rule.
            // Therefore, this state is a winning state because
            // **the previous player** has violated the rule.
            node->predicted_win_rate = 1.0;
            node->predicted_draw_rate = 0.0;
        } else {
            node->predicted_win_rate = 0.0;
            node->predicted_draw_rate = 0.0;
        }

        return;
    }

    // Immediate win by the rule of declaration.
    if (state_api->canDeclare(state)) {
        node->predicted_win_rate = 1.0;
        node->predicted_draw_rate = 0.0;
        return;
    }

    // Evaluation by the neural networks.
    makeFeature(onnx_runtime->input, state, state_config);
    runSession(onnx_runtime);

    // Check repetition.
    node->repetition = state_api->getRepetition(state);
    if (node->repetition == NSHOGI_REPETITION) {
        float draw_value = state_api->getSideToMove(state) == NSHOGI_BLACK
            ? state_api->getBlackDrawValue(state_config)
            : state_api->getWhiteDrawValue(state_config);
        node->predicted_win_rate = draw_value;
        node->predicted_draw_rate = 1.0;
    } else if (node->repetition == NSHOGI_WIN_REPETITION || node->repetition == NSHOGI_SUPERIOR_REPETITION) {
        node->predicted_win_rate = 1.0;
        node->predicted_draw_rate = 0.0;
    } else if (node->repetition == NSHOGI_LOSS_REPETITION || node->repetition == NSHOGI_INFERIOR_REPETITION) {
        node->predicted_win_rate = 0.0;
        node->predicted_draw_rate = 0.0;
    } else { // No repetition.
        node->predicted_win_rate = onnx_runtime->value[0];
        node->predicted_draw_rate = onnx_runtime->draw[0];
    }

    // Normalize policy over the legal moves.
    float legal_policy[600];
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        int move_index = ml_api->moveToIndex(state, child->move);
        legal_policy[i] = onnx_runtime->policy[move_index];
    }
    softmax(legal_policy, node->num_children, 1.0f);

    // Set policy.
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        child->policy_from_parent = legal_policy[i];
    }
}

///
/// @fn backpropagateLeafNode
/// @brief Backpropagate the leaf result to its ancestors.
///
static void backpropagateLeafNode(node_t* node, nshogi_state_t* state) {
    // Indicator for the concidence between the playoer of
    // the leaf node and a node to backpropagate in the ancestors.
    int flip = 0;

    // Leaf results.
    double leaf_win_rate = node->predicted_win_rate;
    double leaf_draw_rate = node->predicted_draw_rate;

    while (1) {
        // If flip == true, flip the leaf value.
        double v = flip ? (1 - leaf_win_rate) : leaf_win_rate;

        // Add leaf values to that in an ancestor.
        node->accumulated_win_rate += v;
        node->accumulated_draw_rate += leaf_draw_rate;
        ++node->visit_count;

        // Go back to its parent node.
        node = node->parent;

        if (node == NULL) {
            break;
        }

        // Reverse the player.
        flip = 1 - flip;
        state_api->undoMove(state);
    }
}

///
/// @fn decideBestChild
/// @brief Decide the best child to play after the search finishes.
///
static node_t* decideBestChild(node_t* node, nshogi_state_t* state, nshogi_state_config_t* state_config, float temperature, double away) {
    // The best child in MCTS is the child with the largest number of visits.
    node_t* best_child = NULL;

    float draw_value = state_api->getSideToMove(state) == NSHOGI_BLACK
        ? state_api->getBlackDrawValue(state_config)
        : state_api->getWhiteDrawValue(state_config);

    int visit_count_max = 0;
    float visit_count_max_policy = 0;
    double best_win_rate = 0.0;
    float win_rates[600];
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        if (child->visit_count == 0) {
            continue;
        }
        double value = child->accumulated_win_rate / child->visit_count;
        double draw = child->accumulated_draw_rate / child->visit_count;
        win_rates[i] = (1.0 - draw) * (1.0 - value) + draw * draw_value;

        if (child->visit_count > visit_count_max) {
            visit_count_max = child->visit_count;
            visit_count_max_policy = child->policy_from_parent;
            best_child = child;
            if (win_rates[i] > best_win_rate) {
                best_win_rate = win_rates[i];
            }
        } else if (child->visit_count == visit_count_max) {
            // If this child has the same visit count as the best one,
            // the child with the higher policy value is chosen.
            if (child->policy_from_parent > visit_count_max_policy) {
                visit_count_max_policy = child->policy_from_parent;
                best_child = child;
                if (win_rates[i] > best_win_rate) {
                    best_win_rate = win_rates[i];
                }
            }
        }
    }

    // If `temperature` is zero, returns the child with
    // the largest number of the visits.
    if (temperature <= 0) {
        return best_child;
    }

    // Find out promising children: a node with
    // win_rate_i that win_rate_i is higher or equal to
    // (win_rate_best - away) is promising.
    int num_candidates = 0;
    node_t* candidate_children[600];
    float visits[600];
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        if (child->visit_count == 0) {
            continue;
        }

        if (win_rates[i] >= best_win_rate - away) {
            candidate_children[num_candidates] = child;
            visits[num_candidates] = child->visit_count;
            ++num_candidates;
        }
    }

    mcts_policy_with_temperature(visits, num_candidates, temperature);

    // Sample a promising child.
    float r = getRandom();
    float s = 0;
    for (int i = 0; i < num_candidates; ++i) {
        s += visits[i];
        if (s >= r) {
            return candidate_children[i];
        }
    }
    return candidate_children[num_candidates - 1];
}

///
/// @fv printThinkLog
/// @brief print thinking log.
///
static void printThinkingLog(double elapsed, tree_t* tree, nshogi_state_t* state, nshogi_state_config_t* state_config) {
    // Output score.
    printf("info time %d score cp %d pv",
            (int)(elapsed * 1000),
            winRateToScore(tree->root->accumulated_win_rate / tree->root->visit_count));

    node_t* node = tree->root;

    // Output principal variation (PV).
    while (1) {
        if (node->num_children == 0) {
            break;
        }

        node_t* best_child = decideBestChild(node, state, state_config, 0, 0);

        if (best_child == NULL) {
            break;
        }

        char* sfen = io_api->moveToSfen(best_child->move);
        printf(" %s", sfen);
        free(sfen);

        node = best_child;
    }
    printf("\n");
    fflush(stdout);
}

///
/// @fn createTree
/// @brief Create a tree_t instance.
///
static tree_t* createTree(void) {
    tree_t* tree = (tree_t*)malloc(sizeof(tree_t));

    tree->root = (node_t*)malloc(sizeof(node_t));
    initializeNode(tree->root);

    return tree;
}

///
/// @fn destroyTree
/// @brief Destroy a tree_t instance.
///
static void destroyTree(tree_t* tree) {
    if (tree->root != NULL) {
        destroyNode(tree->root);
    }

    free(tree);
}

nshogi_move_t startSearch(nshogi_state_t* state, nshogi_state_config_t* state_config, int num_simulation, onnxruntime_t* onnx_runtime, float temperature) {
    time_t start_time = time(NULL);

    // Set a new random seed.
    srand(start_time);

    // If one can win by the declaration rule,
    // immedeately returns the declaration move.
    if (state_api->canDeclare(state)) {
        return nshogi_api->winDeclarationMove();
    }

    // Create a search tree instance.
    tree_t* tree = createTree();

    // Start searching.
    for (int simulation = 0; simulation < num_simulation; ++simulation) {
        // Step 1: select the best leaf node to search.
        node_t* leaf = pickUpLeafNode(tree, state, state_config);

        // Step 2: expand and evaluate the leaf node.
        if (leaf->visit_count == 0) {
            expandNode(state, leaf);
            evaluateNode(state, state_config, leaf, onnx_runtime);
        }

        // Step 3: backpropagate the leaf through the root node.
        backpropagateLeafNode(leaf, state);
    }

    // Print debug output.
    // Format:
    //  [Move sfen]: <policy value>, <win rate value>, <draw value>, <visit count>
    for (int i = 0; i < tree->root->num_children; ++i) {
        node_t* child = tree->root->children[i];

        if (child->visit_count > 0) {
            char* sfen = io_api->moveToSfen(child->move);
            printf("[%-5s]: policy: %.3f, win_rate: %.3f, draw_rate: %.3f, visit_count: %8" PRIu64 "\n",
                    sfen,
                    child->policy_from_parent,
                    1.0 - child->accumulated_win_rate / child->visit_count,
                    child->accumulated_draw_rate / child->visit_count,
                    child->visit_count);
            fflush(stdout);
            free(sfen);
        }
    }

    // Print thinking log.
    time_t end_time = time(NULL);
    double elapsed = difftime(end_time, start_time);
    printThinkingLog(elapsed, tree, state, state_config);

    // Fetch the best move.
    nshogi_move_t best_move = 0;
    node_t* best_child = decideBestChild(tree->root, state, state_config, temperature, 0.01);
    if (best_child != NULL) {
        best_move = best_child->move;
    }

    // Destroy the tree instance.
    destroyTree(tree);

    return best_move;
}
