#ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_THINK
#define NSHOGI_ENGINE_COMMAND_COMMANDS_THINK

#include "../command.h"

namespace nshogi {
namespace engine {
namespace command {
namespace commands {

class Think : public ICommand {
    ~Think();

    CommandType type() const;
};

} // namespace commands
} // namespace command
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMANDS_THINK
