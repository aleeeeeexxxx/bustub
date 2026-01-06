//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// aggregation_executor.cpp
//
// Identification: src/execution/aggregation_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>
#include "common/macros.h"
#include "storage/table/tuple.h"

#include "execution/executors/aggregation_executor.h"

namespace bustub {

/**
 * Construct a new AggregationExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The insert plan to be executed
 * @param child_executor The child executor from which inserted tuples are pulled (may be `nullptr`)
 */
AggregationExecutor::AggregationExecutor(ExecutorContext *exec_ctx, const AggregationPlanNode *plan,
                                         std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      child_executor_(std::move(child_executor)),
      aht_(plan->aggregates_, plan->agg_types_) {}

/** Initialize the aggregation */
void AggregationExecutor::Init() {
  child_executor_->Init();

  Aggregate();
  aht_iterator_ = std::make_unique<SimpleAggregationHashTable::Iterator>(aht_.Begin());
}

/**
 * Yield the next tuple batch from the aggregation.
 * @param[out] tuple_batch The next batch of tuples produced by the aggregation
 * @param[out] rid_batch The next batch of tuple RIDs produced by the aggregation
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if any tuples were produced, `false` if there are no more tuples
 */

auto AggregationExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                               size_t batch_size) -> bool {
  if (*aht_iterator_ == aht_.End()) {
    return false;
  }

  for (size_t i = 0; i < batch_size; i++) {
    if (*aht_iterator_ == aht_.End()) {
      break;
    }

    Tuple tuple{aht_iterator_->Val().aggregates_, &plan_->OutputSchema()};
    tuple_batch->emplace_back(tuple);
    ++(*aht_iterator_);
  }

  return true;
}

/** Do not use or remove this function; otherwise, you will get zero points. */
auto AggregationExecutor::GetChildExecutor() const -> const AbstractExecutor * { return child_executor_.get(); }

auto AggregationExecutor::Aggregate() -> void {
  std::vector<bustub::Tuple> tuples;
  std::vector<bustub::RID> rid_batch;

  while (child_executor_->Next(&tuples, &rid_batch, BUSTUB_BATCH_SIZE)) {
    for (const auto &tuple : tuples) {
      AggregateKey key = MakeAggregateKey(&tuple);
      AggregateValue val = MakeAggregateValue(&tuple);
      aht_.InsertCombine(key, val);
    }

    tuples.clear();
    rid_batch.clear();
  }
}

}  // namespace bustub
