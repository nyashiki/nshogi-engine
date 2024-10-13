#ifndef NSHOGI_ENGINE_COMMAND_COMMAND_H
#define NSHOGI_ENGINE_COMMAND_COMMAND_H

namespace nshogi {
namespace engine {
namespace command {

enum class CommandType {
    CT_Noop,
    CT_Config,
    CT_GetReady,
    CT_SetPosition,
    CT_Think,
};

class ICommand {
 public:
    virtual ~ICommand() {
    }

    virtual CommandType type() const = 0;

};

} // namespace command
} // namespace engine
} // namespace nshogi


#endif // #ifndef NSHOGI_ENGINE_COMMAND_COMMAND_H
