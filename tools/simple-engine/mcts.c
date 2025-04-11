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

static void softmax(float* x, int num) {
    float max_value = x[0];
    for (int i = 1; i < num; ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < num; ++i) {
        x[i] = exp(x[i] - max_value);
        sum += x[i];
    }

    for (int i = 0; i < num; ++i) {
        x[i] /= sum;
    }
}

static int winRateToScore(double x) {
    if (x == 0) {
        return -9999;
    } else if (x == 1) {
        return 9999;
    }

    return (int)(-600 * log(1 / x - 1));
}

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

static void destroyNode(node_t* node) {
    if (node->num_children != 0) {
        for (int i = 0; i < node->num_children; ++i) {
            destroyNode(node->children[i]);
        }
        free(node->children);
    }

    free(node);
}

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

static node_t* selectChildToSearch(node_t* node) {
    node_t* ucb_best_child = NULL;
    double ucb_best_value = 0;

    double c_base = 19652;
    double c_init = 1.25;
    double c = log((double)((1 + node->visit_count + c_base) / c_base)) + c_init;
    double n_sqrt = sqrt((double)node->visit_count);
    double c_n_sqrt = c * n_sqrt;

    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];

        double q = 0;

        if (child->visit_count > 0) {
            double child_win_rate = child->accumulated_win_rate / child->visit_count;
            double child_draw_rate = child->accumulated_draw_rate / child->visit_count;
            q = (1.0 - child_draw_rate) * (1.0 - child_win_rate) + child_draw_rate * 0.5;
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

static node_t* pickUpLeafNode(tree_t* tree, nshogi_state_t* state) {
    node_t* node = tree->root;

    while (1) {
        if (node->visit_count == 0) {
            break;
        }

        if (node->num_children == 0) {
            break;
        }

        if (node != tree->root && node->repetition != NSHOGI_NO_REPETITION) {
            break;
        }

        node = selectChildToSearch(node);
        state_api->doMove(state, node->move);
    }

    return node;
}

static void evaluateNode(nshogi_state_t* state, nshogi_state_config_t* state_config, node_t* node, onnxruntime_t* onnx_runtime) {
    // Checkmate.
    if (node->num_children == 0) {
        if (nshogi_api->isDroppingPawn(node->move)) {
            // Checkmate by a dropping pawn, which is prohibited by the rule.
            node->predicted_win_rate = 1.0;
            node->predicted_draw_rate = 0.0;
        } else {
            node->predicted_win_rate = 0.0;
            node->predicted_draw_rate = 0.0;
        }

        return;
    }

    if (state_api->canDeclare(state)) {
        node->predicted_win_rate = 1.0;
        node->predicted_draw_rate = 0.0;
        return;
    }

    makeFeature(onnx_runtime->input, state, state_config);
    runSession(onnx_runtime);

    // Check repetition.
    node->repetition = state_api->getRepetition(state);
    if (node->repetition == NSHOGI_REPETITION) {
        node->predicted_win_rate = 0.5;
        node->predicted_draw_rate = 1.0;
    } else if (node->repetition == NSHOGI_WIN_REPETITION || node->repetition == NSHOGI_SUPERIOR_REPETITION) {
        node->predicted_win_rate = 1.0;
        node->predicted_draw_rate = 0.0;
    } else if (node->repetition == NSHOGI_LOSS_REPETITION || node->repetition == NSHOGI_INFERIOR_REPETITION) {
        node->predicted_win_rate = 0.0;
        node->predicted_draw_rate = 0.0;
    } else {
        node->predicted_win_rate = onnx_runtime->value[0];
        node->predicted_draw_rate = onnx_runtime->draw[0];
    }

    // Normalize policy over the legal moves.
    float legal_policy[600];
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        int move_index = nshogi_api->moveToIndex(state, child->move);
        legal_policy[i] = onnx_runtime->policy[move_index];
    }
    softmax(legal_policy, node->num_children);

    // Set policy.
    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];
        child->policy_from_parent = legal_policy[i];
    }
}

static void backpropagateLeafNode(node_t* node, nshogi_state_t* state) {
    int flip = 0;

    double leaf_win_rate = node->predicted_win_rate;
    double leaf_draw_rate = node->predicted_draw_rate;

    while (1) {
        double v = flip ? (1 - leaf_win_rate) : leaf_win_rate;

        node->accumulated_win_rate += v;
        node->accumulated_draw_rate += leaf_draw_rate;
        ++node->visit_count;

        node = node->parent;

        if (node == NULL) {
            break;
        }

        flip = 1 - flip;
        state_api->undoMove(state);
    }
}

static node_t* decideBestChild(node_t* node) {
    node_t* best_child = NULL;

    int visit_count_max = 0;
    float visit_count_max_policy = 0;

    for (int i = 0; i < node->num_children; ++i) {
        node_t* child = node->children[i];

        if (child->visit_count > visit_count_max) {
            visit_count_max = child->visit_count;
            visit_count_max_policy = child->policy_from_parent;
            best_child = child;
        } else if (child->visit_count == visit_count_max) {
            if (child->policy_from_parent > visit_count_max_policy) {
                visit_count_max_policy = child->policy_from_parent;
                best_child = child;
            }
        }
    }

    return best_child;
}

static void printThinkLog(tree_t* tree) {
    printf("info score cp %d pv",
            winRateToScore(tree->root->accumulated_win_rate / tree->root->visit_count));

    node_t* node = tree->root;

    while (1) {
        if (node->num_children == 0) {
            break;
        }

        node_t* best_child = decideBestChild(node);
        char* sfen = io_api->moveToSfen(best_child->move);
        printf(" %s", sfen);
        free(sfen);

        node = best_child;
    }
    printf("\n");
    fflush(stdout);
}

static tree_t* createTree(void) {
    tree_t* tree = (tree_t*)malloc(sizeof(tree_t));

    tree->root = (node_t*)malloc(sizeof(node_t));
    initializeNode(tree->root);

    return tree;
}

static void destroyTree(tree_t* tree) {
    if (tree->root != NULL) {
        destroyNode(tree->root);
    }

    free(tree);
}

nshogi_move_t startSearch(nshogi_state_t* state, nshogi_state_config_t* state_config, int num_simulation, onnxruntime_t* onnx_runtime) {
    if (state_api->canDeclare(state)) {
        return nshogi_api->winDeclarationMove();
    }

    tree_t* tree = createTree();

    for (int simulation = 0; simulation < num_simulation; ++simulation) {
        node_t* leaf = pickUpLeafNode(tree, state);

        if (leaf->visit_count == 0) {
            expandNode(state, leaf);
            evaluateNode(state, state_config, leaf, onnx_runtime);
        }

        backpropagateLeafNode(leaf, state);
    }

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

    printThinkLog(tree);
    nshogi_move_t best_move = 0;
    node_t* best_child = decideBestChild(tree->root);
    if (best_child != NULL) {
        best_move = best_child->move;
    }

    destroyTree(tree);

    return best_move;
}
