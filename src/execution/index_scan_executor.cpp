//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// index_scan_executor.cpp
//
// Identification: src/execution/index_scan_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/index_scan_executor.h"
#include <memory>
#include "common/macros.h"
#include "execution/expressions/constant_value_expression.h"
#include "storage/table/tuple.h"

namespace bustub {

/**
 * Creates a new index scan executor.
 * @param exec_ctx the executor context
 * @param plan the index scan plan to be executed
 */
IndexScanExecutor::IndexScanExecutor(ExecutorContext *exec_ctx, const IndexScanPlanNode *plan)
    : AbstractExecutor(exec_ctx), plan_(plan) {}

void IndexScanExecutor::Init() {
  cur_idx_ = 0;

  if (!plan_->pred_keys_.empty()) {
    return;
  }

  InitItr();
}

auto IndexScanExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                             size_t batch_size) -> bool {
  if (table_itr_ != nullptr) {
    return NextItr(tuple_batch, rid_batch, batch_size);
  }
  return NextScan(tuple_batch, rid_batch, batch_size);
}

auto IndexScanExecutor::InitItr() -> void {
  auto index = exec_ctx_->GetCatalog()->GetIndex(plan_->GetIndexOid());
  auto btree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index->index_.get());

  table_itr_ = std::make_unique<BPlusTreeIndexIteratorForTwoIntegerColumn>(btree_index->GetBeginIterator());
};

auto IndexScanExecutor::NextItr(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                size_t batch_size) -> bool {
  if (table_itr_->IsEnd()) {
    return false;
  }

  for (size_t i = 0; i < batch_size;) {
    if (table_itr_->IsEnd()) {
      break;
    }

    auto [key, rid] = **table_itr_;
    ++(*table_itr_);

    auto [meta, tuple] = exec_ctx_->GetCatalog()->GetTable(plan_->table_oid_)->table_->GetTuple(rid);
    if (meta.is_deleted_) {
      continue;
    }

    tuple_batch->emplace_back(tuple);
    rid_batch->emplace_back(rid);
    i++;
  }

  return true;
}

auto IndexScanExecutor::NextScan(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                 size_t batch_size) -> bool {
  if (cur_idx_ >= plan_->pred_keys_.size()) {
    return false;
  }

  auto index = exec_ctx_->GetCatalog()->GetIndex(plan_->GetIndexOid());
  BUSTUB_ENSURE(index.get() != nullptr, "Index not found");

  for (; tuple_batch->size() < batch_size && cur_idx_ < plan_->pred_keys_.size();) {
    auto cur = plan_->pred_keys_[cur_idx_];
    cur_idx_++;

    auto expr = dynamic_cast<ConstantValueExpression *>(cur.get());

    std::vector<Value> key_values{expr->val_};
    Tuple key{key_values, index->index_->GetKeySchema()};

    std::vector<bustub::RID> rids;

    index->index_->ScanKey(key, &rids, exec_ctx_->GetTransaction());
    if (rids.empty()) {
      continue;
    }

    for (const auto &rid : rids) {
      auto [meta, tuple] = exec_ctx_->GetCatalog()->GetTable(plan_->table_oid_)->table_->GetTuple(rid);
      if (meta.is_deleted_) {
        continue;
      }

      tuple_batch->emplace_back(tuple);
      rid_batch->emplace_back(rid);
    }
  }

  return true;
};

}  // namespace bustub
