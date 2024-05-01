#include "bench.h"

#include "batchsize.h"

#include <nshogi/core/utils.h>

#include <iostream>

namespace nshogi {
namespace engine {
namespace bench {


void mainLoop() {
    std::cout << "Entering bench mode." << std::endl;

    std::string Line;
    while (std::getline(std::cin, Line)) {
        if (Line.size() == 0) {
            continue;
        }

        const auto Splitted = nshogi::core::utils::split(Line, ' ');

        if (Splitted[0] == "BatchSize") {
            std::string WeightPath = "./res/model.onnx";
            std::size_t Repeat = 1000;

            if (Splitted.size() >= 2) {
                WeightPath = Splitted[1];
            }

            if (Splitted.size() >= 3) {
                Repeat = std::stoul(Splitted[2]);
            }

            benchBatchSize(WeightPath.c_str(), Repeat);
        } else {
            std::cout << "Unknown command `" << Splitted[0] << "`." << std::endl;
        }
    }
}

} // namespace bench
} // namespace engine
} // namespace nshogi
