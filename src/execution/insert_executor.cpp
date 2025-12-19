//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <memory>
#include "common/macros.h"
#include "storage/table/tuple.h"
#include "type/value_factory.h"

#include "execution/executors/insert_executor.h"

namespace bustub {

/**
 * Construct a new InsertExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The insert plan to be executed
 * @param child_executor The child executor from which inserted tuples are pulled
 */
InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

/** Initialize the insert */
void InsertExecutor::Init() { child_executor_->Init(); }

/**
 * Yield the number of rows inserted into the table.
 * @param[out] tuple_batch The tuple batch with one integer indicating the number of rows inserted into the table
 * @param[out] rid_batch The next tuple RID batch produced by the insert (ignore, not used)
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 *
 * NOTE: InsertExecutor::Next() does not use the `rid_batch` out-parameter.
 * NOTE: InsertExecutor::Next() returns true with the number of inserted rows produced only once.
 */
auto InsertExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                          size_t batch_size) -> bool {
  auto oid = plan_->GetTableOid();
  auto &table = exec_ctx_->GetCatalog()->GetTable(oid)->table_;

  auto tupple_meta = TupleMeta{0, false};

  std::vector<bustub::Tuple> child_tuples;
  std::vector<bustub::RID> child_rids;

  if (!child_executor_->Next(&child_tuples, &child_rids, batch_size)) {
    return false;
  }

  tuple_batch->push_back(GenerateResultTuple(child_tuples.size()));

  for (const auto &tuple : child_tuples) {
    auto rid = table->InsertTuple(tupple_meta, tuple, exec_ctx_->GetLockManager(), exec_ctx_->GetTransaction(), oid);
    if (!rid.has_value()) {
      throw bustub::Exception("InsertExecutor: failed to insert tuple");
    }
  }

  return true;
}

auto InsertExecutor::GenerateResultTuple(size_t value) -> Tuple {
  std::vector<Value> values;
  values.emplace_back(ValueFactory::GetIntegerValue(static_cast<int32_t>(value)));
  return Tuple(values, &GetOutputSchema());
}

}  // namespace bustub
