#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_STOP
#define NSHOGI_ENGINE_COMMAND_COMMANDS_STOP

#include "../command.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class Stop : public ICommand {
 public:
    CommandType type() const;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_STOP
