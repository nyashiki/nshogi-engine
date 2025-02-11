# nshogi-engine: a USI shogi engine

nshogi-engine is a USI shogi engine built on the top of (nshogi)[https://github.com/nyashiki/nshogi] library.
The engine implements an AlphaZero-style Monte-Carlo Tree Search (MCTS) algorithm, using multi-threading for search.
In addition, this project includes a module for generating self-play data used for reinforcement learning in the AlphaZero style.

## How to Build

### Prerequisites

Ensure that the following are installed:

- `make`
- `g++` or `clang++`
- **For the TensorRT executor:** CUDA and TensorRT
- **For testing:** googletest

### Build Instructions

Open your terminal and run:

```bash
make [CUDA_ENABLED=1] EXECUTOR=(tensorrt|random) engine
```

Note: If you are using the TensorRT executor (EXECUTOR=tensorrt), you must specify `CUDA_ENABLED=1`.

The USI engine binary is built in `./build` directory.

## License

This repository is released under the MIT License.

For details about licenses of third-party dependencies, please see [LICENSE-THIRD-PARTY.md](./LICENSE-THIRD-PARTY.md).
