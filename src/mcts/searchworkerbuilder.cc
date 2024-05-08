#include "searchworkerbuilder.h"

namespace nshogi {
namespace engine {
namespace mcts {

SearchWorkerBuilder::SearchWorkerBuilder()
    : BatchSize(0)
    , PEvaluator(nullptr)
    , PCheckmateSearcher(nullptr)
    , PMutexPool(nullptr)
    , PEvalCache(nullptr) {
}

std::unique_ptr<SearchWorker> SearchWorkerBuilder::build() const {
    return std::make_unique<SearchWorker>(
                        BatchSize,
                        PEvaluator,
                        PCheckmateSearcher,
                        PMutexPool,
                        PEvalCache);
}

SearchWorkerBuilder& SearchWorkerBuilder::setBatchSize(std::size_t Size) {
    BatchSize = Size;
    return *this;
}

SearchWorkerBuilder& SearchWorkerBuilder::setEvaluator(evaluate::Evaluator* Evaluator) {
    PEvaluator = Evaluator;
    return *this;
}

SearchWorkerBuilder& SearchWorkerBuilder::setCheckmateSearcher(CheckmateSearcher* Searcher) {
    PCheckmateSearcher = Searcher;
    return *this;
}

SearchWorkerBuilder& SearchWorkerBuilder::setMutexPool(MutexPool<lock::SpinLock>* MtxPool) {
    PMutexPool = MtxPool;
    return *this;
}

SearchWorkerBuilder& SearchWorkerBuilder::setEvalCache(EvalCache* ECache) {
    PEvalCache = ECache;
    return *this;
}

} // namespace mcts
} // namespace engine
} // namespace nshogi
