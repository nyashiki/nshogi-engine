#include "setposition.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

SetPosition::SetPosition(const char* Sfen)
    : _Sfen(Sfen) {
}

CommandType SetPosition::type() const {
    return CommandType::CT_SetPosition;
}

const char* SetPosition::sfen() const {
    return _Sfen.c_str();
}

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi
