#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_NOOP
#define NSHOGI_ENGINE_COMMAND_COMMANDS_NOOP

#include "../command.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class Noop : public ICommand {
    CommandType type() const;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_NOOP
