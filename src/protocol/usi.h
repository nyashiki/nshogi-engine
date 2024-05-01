#ifndef NSHOGI_ENGINE_PROTOCOL_USI_H
#define NSHOGI_ENGINE_PROTOCOL_USI_H

namespace nshogi {
namespace engine {
namespace protocol {
namespace usi {

static constexpr const char* USIName = "nshogi-engine";
static constexpr const char* USIAuthor = "nyashiki";

void mainLoop();

} // namespace usi
} // namespace protocol
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_PROTOCOL_USI_H
