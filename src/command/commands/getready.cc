#include "getready.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

GetReady::~GetReady() {
}

CommandType GetReady::type() const {
    return CommandType::CT_GetReady;
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
