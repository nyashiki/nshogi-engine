//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include "nshogi.h"

nshogi_api_t* nshogi_api;
nshogi_state_api_t* state_api;
nshogi_ml_api_t* ml_api;
nshogi_io_api_t* io_api;

void initializeNShogi(void) {
    nshogi_api = nshogiApi();
    state_api = nshogi_api->stateApi();
    ml_api = nshogi_api->mlApi();
    io_api = nshogi_api->ioApi();
}
