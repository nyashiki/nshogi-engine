#ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKERBUILDER_H
#define NSHOGI_ENGINE_MCTS_SEARCHWORKERBUILDER_H

#include "checkmatesearcher.h"
#include "evalcache.h"
#include "mutexpool.h"
#include "searchworker.h"
#include "../evaluate/evaluator.h"

#include <memory>
#include <cinttypes>

namespace nshogi {
namespace engine {
namespace mcts {

class SearchWorkerBuilder {
 public:
    SearchWorkerBuilder();
    std::unique_ptr<SearchWorker> build() const;

    SearchWorkerBuilder& setBatchSize(std::size_t Size);
    SearchWorkerBuilder& setEvaluator(evaluate::Evaluator* Evaluator);
    SearchWorkerBuilder& setCheckmateSearcher(CheckmateSearcher* Searcher);
    SearchWorkerBuilder& setMutexPool(MutexPool* MtxPool);
    SearchWorkerBuilder& setEvalCache(EvalCache* ECache);

 private:
    std::size_t BatchSize;
    evaluate::Evaluator* PEvaluator;
    CheckmateSearcher* PCheckmateSearcher;
    MutexPool* PMutexPool;
    EvalCache* PEvalCache;
};

} // namespace mcts
} // namespace engine
} // namespace nshogi

#endif // #ifndef NSHOGI_ENGINE_MCTS_SEARCHWORKERBUILDER_H
