#include "stop.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

CommandType Stop::type() const {
    return CommandType::CT_Stop;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
