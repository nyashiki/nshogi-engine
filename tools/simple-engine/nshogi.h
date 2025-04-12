//
// Copyright (c) 2025 @nyashiki
//
// This software is licensed under the MIT license.
// For details, see the LICENSE file in the root of this repository.
//
// SPDX-License-Identifier: MIT
//

#include <nshogi/c_api.h>

/// @brief nshogi API entry point.
extern nshogi_api_t* nshogi_api;

/// @brief nshogi state API entry point.
extern nshogi_state_api_t* state_api;

/// @brief nshogi ml API entry point.
extern nshogi_ml_api_t* ml_api;

/// @brief nshogi io API entry point.
extern nshogi_io_api_t* io_api;

///
/// @fn
/// @brief Initialize nshogi library.
///
void initializeNShogi(void);
