//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nested_index_join_executor.cpp
//
// Identification: src/execution/nested_index_join_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/nested_index_join_executor.h"
#include "common/macros.h"
#include "common/rid.h"

namespace bustub {

/**
 * Creates a new nested index join executor.
 * @param exec_ctx the context that the nested index join should be performed in
 * @param plan the nested index join plan to be executed
 * @param child_executor the outer table
 */
NestedIndexJoinExecutor::NestedIndexJoinExecutor(ExecutorContext *exec_ctx, const NestedIndexJoinPlanNode *plan,
                                                 std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {
  if (plan->GetJoinType() != JoinType::LEFT && plan->GetJoinType() != JoinType::INNER) {
    // Note for Spring 2025: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
}

void NestedIndexJoinExecutor::Init() { child_executor_->Init(); }

auto NestedIndexJoinExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                   size_t batch_size) -> bool {
  tuple_batch->clear();
  rid_batch->clear();

  while (tuple_batch->size() < batch_size) {
    if (cache_.Empty()) {
      cache_.Reset();

      std::vector<RID> rids;
      if (!child_executor_->Next(cache_.Raw(), &rids, batch_size)) {
        break;
      }
    }

    auto left = cache_.Pop();
    std::vector<RID> rids;

    auto right = plan_->KeyPredicate()->Evaluate(left, child_executor_->GetOutputSchema());
    auto right_schema = plan_->InnerTableSchema();
    auto index_schema = exec_ctx_->GetCatalog()->GetIndex(plan_->index_oid_)->index_->GetKeySchema();
    auto key = Tuple{{right}, index_schema};

    exec_ctx_->GetCatalog()->GetIndex(plan_->index_oid_)->index_->ScanKey(key, &rids, exec_ctx_->GetTransaction());

    if (!rids.empty()) {
      BUSTUB_ENSURE(rids.size() == 1, "Only support index with unique key for now.");
      auto inner_table = exec_ctx_->GetCatalog()->GetTable(plan_->GetInnerTableOid());
      auto [meta, __right_tuple] = inner_table->table_->GetTuple(rids[0]);
      if (!meta.is_deleted_) {
        tuple_batch->emplace_back(CreateMergedTuple(*left, child_executor_->GetOutputSchema(), &__right_tuple,
                                                    right_schema, GetOutputSchema()));
        continue;
      }
    }

    if (plan_->GetJoinType() == JoinType::LEFT) {
      tuple_batch->emplace_back(
          CreateMergedTuple(*left, child_executor_->GetOutputSchema(), nullptr, right_schema, GetOutputSchema()));
    }
  }

  for (size_t i = 0; i < tuple_batch->size(); i++) {
    rid_batch->push_back(RID{});
  }

  return tuple_batch->size() > 0;
}

}  // namespace bustub
