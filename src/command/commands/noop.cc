#include "noop.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

CommandType Noop::type() const {
    return CommandType::CT_Noop;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
