#include "allocator.h"

#if defined(USE_TLSF_ALLOCATOR)

#include "tlsf.h"

#else

#include "default.h"

#endif

#include "fixed_allocator.h"
#include "segregated_free_list.h"
#include "default.h"
#include "../mcts/node.h"
#include "../mcts/edge.h"

namespace nshogi {
namespace engine {
namespace allocator {

Allocator* createEdgeAllocator() {
    return new SegregatedFreeListAllocator();
}

Allocator* createNodeAllocator() {
    return new FixedAllocator<sizeof(mcts::Node)>();
}

} // nameespace allocator
} // namespace engine
} // namespace nshogi
