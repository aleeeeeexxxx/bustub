//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seq_scan_executor.cpp
//
// Identification: src/execution/seq_scan_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/seq_scan_executor.h"
#include <memory>
#include "common/macros.h"
#include "storage/table/table_iterator.h"

namespace bustub {

/**
 * Construct a new SeqScanExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The sequential scan plan to be executed
 */
SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan)
    : AbstractExecutor(exec_ctx), plan_(plan) {}

/** Initialize the sequential scan */
void SeqScanExecutor::Init() {
  itr_ =
      std::make_unique<TableIterator>(exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid())->table_->MakeIterator());
}

/**
 * Yield the next tuple batch from the seq scan.
 * @param[out] tuple_batch The next tuple batch produced by the scan
 * @param[out] rid_batch The next tuple RID batch produced by the scan
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 */
auto SeqScanExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                           size_t batch_size) -> bool {
  for (size_t i = 0; i < batch_size;) {
    if (itr_->IsEnd()) {
      break;
    }

    auto [meta, tuple] = itr_->GetTuple();
    ++(*itr_);

    if (meta.is_deleted_) {
      continue;
    }

    if (plan_->filter_predicate_ != nullptr) {
      auto value =
          plan_->filter_predicate_->Evaluate(&tuple, exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid())->schema_);
      if (value.IsNull() || !value.GetAs<bool>()) {
        continue;
      }
    }

    tuple_batch->push_back(tuple);
    rid_batch->push_back(tuple.GetRid());

    ++i;
  }

  return !tuple_batch->empty();
}

}  // namespace bustub
