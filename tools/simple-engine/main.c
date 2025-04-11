//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "onnxruntime.h"
#include "mcts.h"
#include "nshogi.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    initializeNShogi();

    onnxruntime_t* onnx_runtime = NULL;

    ssize_t read;
    size_t len = 0;
    char* line = NULL;

    nshogi_state_t* state = NULL;
    nshogi_state_config_t* state_config = state_api->createStateConfig();

    while ((read = getline(&line, &len, stdin)) != -1) {
        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
        }
        if (strncmp(line, "usi", 3) == 0) {
            printf("id name simple-engine\n");
            printf("id author nyashiki\n");
            printf("usiok\n");
            fflush(stdout);
        } else if(strncmp(line, "isready", 7) == 0) {
            // Initialize onnxruntime.
            onnx_runtime = createOnnxRuntimeInstance("model.onnx");
            printf("readyok\n");
            fflush(stdout);
        } else if (strncmp(line, "usinewgame", 10) == 0) {

        } else if (strncmp(line, "position", 8) == 0) {
            // Set the state.
            if (state != NULL) {
                state_api->destroyState(state);
            }
            if (strncmp(line + 9, "sfen", 4) == 0) {
                printf("DEBUG SFEN:%s\n", line + 14);
                state = io_api->createStateFromSfen(line + 14);
            } else {
                state = io_api->createStateFromSfen(line + 9);
            }
        } else if (strncmp(line, "go", 2) == 0) {
            nshogi_move_t best_move = startSearch(state, state_config, 800, onnx_runtime);
            char* sfen = io_api->moveToSfen(best_move);
            printf("bestmove %s\n", sfen);
            fflush(stdout);
            free(sfen);
        } else if (strncmp(line, "debug", 5) == 0) {
            nshogi_move_t moves[600];
            int move_count = state_api->generateMoves(state, 1, moves);
            printf("move_count: %d\n", move_count);
            printf("moves: ");
            for (int i = 0; i < move_count; ++i) {
                char* sfen = io_api->moveToSfen(moves[i]);
                printf("%s, ", sfen);
                free(sfen);
            }
            printf("\n");
            fflush(stdout);
        } else if (strncmp(line, "exit", 4) == 0 || strncmp(line, "quit", 4) == 0) {
            break;
        } else {
            printf("Unknown command: %s\n", line);
            fflush(stdout);
        }
    }

    free(line);

    if (state != NULL) {
        state_api->destroyState(state);
    }
    state_api->destroyStateConfig(state_config);

    if (onnx_runtime != NULL) {
        destroyOnnxRuntimeInstance(onnx_runtime);
    }

    return 0;
}
