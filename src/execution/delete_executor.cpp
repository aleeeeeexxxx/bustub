//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// delete_executor.cpp
//
// Identification: src/execution/delete_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>
#include "common/macros.h"

#include "execution/executors/delete_executor.h"

namespace bustub {

/**
 * Construct a new DeleteExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The delete plan to be executed
 * @param child_executor The child executor that feeds the delete
 */
DeleteExecutor::DeleteExecutor(ExecutorContext *exec_ctx, const DeletePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

/** Initialize the delete */
void DeleteExecutor::Init() { child_executor_->Init(); }

/**
 * Yield the number of rows deleted from the table.
 * @param[out] tuple_batch The tuple batch with one integer indicating the number of rows deleted from the table
 * @param[out] rid_batch The next tuple RID batch produced by the delete (ignore, not used)
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 *
 * NOTE: DeleteExecutor::Next() does not use the `rid_batch` out-parameter.
 * NOTE: DeleteExecutor::Next() returns true with the number of deleted rows produced only once.
 */
auto DeleteExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                          size_t batch_size) -> bool {
  if (end_) {
    return false;
  }
  end_ = true;

  auto oid = plan_->GetTableOid();
  auto table_info = exec_ctx_->GetCatalog()->GetTable(oid);
  auto indices = exec_ctx_->GetCatalog()->GetTableIndexes(table_info->name_);

  std::vector<bustub::Tuple> child_tuples;
  std::vector<bustub::RID> child_rids;

  size_t deleted = 0;

  while (child_executor_->Next(&child_tuples, &child_rids, batch_size)) {
    for (size_t i = 0; i < child_tuples.size(); ++i) {
      const auto &old_tuple = child_tuples[i];
      const auto &rid = child_rids[i];

      table_info->table_->UpdateTupleMeta({0, true}, rid);

      for (auto &index : indices) {
        auto attrs = index->index_->GetKeyAttrs();

        index->index_->DeleteEntry(old_tuple.KeyFromTuple(table_info->schema_, index->key_schema_, attrs), rid,
                                   exec_ctx_->GetTransaction());
      }
    }

    deleted += child_tuples.size();

    child_tuples.clear();
    child_rids.clear();
  }

  tuple_batch->push_back(GenerateResultTuple(deleted));
  return true;
}

}  // namespace bustub
