//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// update_executor.cpp
//
// Identification: src/execution/update_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>
#include "common/macros.h"

#include "execution/executors/update_executor.h"

namespace bustub {

/**
 * Construct a new UpdateExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The update plan to be executed
 * @param child_executor The child executor that feeds the update
 */
UpdateExecutor::UpdateExecutor(ExecutorContext *exec_ctx, const UpdatePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

/** Initialize the update */
void UpdateExecutor::Init() { child_executor_->Init(); }

/**
 * Yield the number of rows updated in the table.
 * @param[out] tuple_batch The tuple batch with one integer indicating the number of rows updated in the table
 * @param[out] rid_batch The next tuple RID batch produced by the update (ignore, not used)
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 *
 * NOTE: UpdateExecutor::Next() does not use the `rid_batch` out-parameter.
 * NOTE: UpdateExecutor::Next() returns true with the number of updated rows produced only once.
 */
auto UpdateExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                          size_t batch_size) -> bool {
  tuple_batch->clear();
  rid_batch->clear();

  if (end_) {
    return false;
  }
  end_ = true;

  std::vector<bustub::Tuple> child_tuples;
  std::vector<bustub::RID> child_rids;

  auto oid = plan_->GetTableOid();
  auto table_info = exec_ctx_->GetCatalog()->GetTable(oid);
  auto indices = exec_ctx_->GetCatalog()->GetTableIndexes(table_info->name_);

  size_t updated = 0;

  while (child_executor_->Next(&child_tuples, &child_rids, batch_size)) {
    for (size_t i = 0; i < child_tuples.size(); ++i) {
      const auto &old_tuple = child_tuples[i];
      const auto &rid = child_rids[i];

      table_info->table_->UpdateTupleMeta({0, true}, rid);

      auto updated = GetUpdatedTuple(old_tuple);
      auto new_rid = table_info->table_->InsertTuple({0, false}, updated);
      BUSTUB_ENSURE(new_rid.has_value(), "Failed to insert new tuple");

      UpdateIndex(old_tuple, updated, rid, new_rid.value(), indices, table_info->schema_);
    }

    updated += child_tuples.size();

    child_tuples.clear();
    child_rids.clear();
  }

  tuple_batch->push_back(GenerateResultTuple(updated));
  rid_batch->push_back(RID{});
  return true;
}

auto UpdateExecutor::GetUpdatedTuple(const Tuple &old_tuple) -> Tuple {
  // return Tuple(updated_values, &plan_->OutputSchema());
  std::vector<Value> values;
  auto schema = child_executor_->GetOutputSchema();

  for (auto &expression : plan_->target_expressions_) {
    auto col = expression->Evaluate(&old_tuple, schema);
    values.push_back(col);
  }

  return Tuple{values, &schema};
}

auto UpdateExecutor::UpdateIndex(const Tuple &old_tuple, const Tuple &new_tuple, const RID &old_rid, const RID &new_rid,
                                 const std::vector<std::shared_ptr<IndexInfo>> &indices, const Schema &schema) -> void {
  for (auto &index : indices) {
    index->index_->DeleteEntry(old_tuple.KeyFromTuple(schema, index->key_schema_, index->index_->GetKeyAttrs()),
                               old_rid, exec_ctx_->GetTransaction());

    index->index_->InsertEntry(new_tuple.KeyFromTuple(schema, index->key_schema_, index->index_->GetKeyAttrs()),
                               new_rid, exec_ctx_->GetTransaction());
  }
}

}  // namespace bustub
