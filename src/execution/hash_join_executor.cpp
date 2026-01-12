//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// hash_join_executor.cpp
//
// Identification: src/execution/hash_join_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/hash_join_executor.h"
#include <functional>
#include <optional>
#include <vector>
#include "common/macros.h"
#include "execution/plans/aggregation_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

/**
 * Construct a new HashJoinExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The HashJoin join plan to be executed
 * @param left_child The child executor that produces tuples for the left side of join
 * @param right_child The child executor that produces tuples for the right side of join
 */
HashJoinExecutor::HashJoinExecutor(ExecutorContext *exec_ctx, const HashJoinPlanNode *plan,
                                   std::unique_ptr<AbstractExecutor> &&left_child,
                                   std::unique_ptr<AbstractExecutor> &&right_child)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      left_child_(std::move(left_child)),
      right_child_(std::move(right_child)) {
  if (plan->GetJoinType() != JoinType::LEFT && plan->GetJoinType() != JoinType::INNER) {
    // Note for Spring 2025: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
}

/** Initialize the join */
void HashJoinExecutor::Init() {
  left_child_->Init();
  right_child_->Init();

  LoadRightTuples();
  joined_tuples_.Reset();
}

/**
 * Yield the next tuple batch from the hash join.
 * @param[out] tuple_batch The next tuple batch produced by the hash join
 * @param[out] rid_batch The next tuple RID batch produced by the hash join
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 */
auto HashJoinExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                            size_t batch_size) -> bool {
  tuple_batch->clear();
  rid_batch->clear();

  while (tuple_batch->size() < batch_size) {
    if (joined_tuples_.Empty()) {
      std::vector<Tuple> tuples;
      std::vector<RID> rids;

      if (!left_child_->Next(&tuples, &rids, 1)) {
        break;
      }

      auto left_tuple = tuples[0];
      Join(&left_tuple);

      continue;
    }

    auto joined_tuple = joined_tuples_.Pop();
    tuple_batch->push_back(*joined_tuple);
  }

  for (size_t i = 0; i < tuple_batch->size(); i++) {
    rid_batch->push_back(RID{});
  }

  return !tuple_batch->empty();
}

auto HashJoinExecutor::Join(Tuple *left_tuple) -> void {
  joined_tuples_.Reset();

  auto left_key = GetJoinKey(left_tuple, plan_->left_key_expressions_);

  auto right_iter = right_tuples_.find(left_key);
  if (right_iter != right_tuples_.end()) {
    auto &right_tuples = right_iter->second;

    for (auto &right_tuple : right_tuples) {
      auto right_key = GetJoinKey(&right_tuple, plan_->right_key_expressions_);

      if (left_key == right_key) {
        joined_tuples_.Push(CreateMergedTuple(*left_tuple, left_child_->GetOutputSchema(), &right_tuple,
                                              right_child_->GetOutputSchema(), plan_->OutputSchema()));
      }
    }
  }

  if (joined_tuples_.Empty() && plan_->GetJoinType() == JoinType::LEFT) {
    joined_tuples_.Push(CreateMergedTuple(*left_tuple, left_child_->GetOutputSchema(), nullptr,
                                          right_child_->GetOutputSchema(), plan_->OutputSchema()));
  }
}

auto HashJoinExecutor::LoadRightTuples() -> void {
  std::vector<Tuple> tuples;
  std::vector<RID> rids;

  while (right_child_->Next(&tuples, &rids, BUSTUB_BATCH_SIZE)) {
    for (auto &tuple : tuples) {
      auto key = GetJoinKey(&tuple, plan_->right_key_expressions_);

      auto itr = right_tuples_.find(key);
      if (itr != right_tuples_.end()) {
        itr->second.push_back(tuple);
      } else {
        right_tuples_.insert({key, {tuple}});
      }
    }

    tuples.clear();
    rids.clear();
  }
}

auto HashJoinExecutor::GetJoinKey(Tuple *tuple, const std::vector<AbstractExpressionRef> &key_exprs) -> AggregateKey {
  AggregateKey key;
  for (const auto &expr : key_exprs) {
    key.group_bys_.push_back(expr->Evaluate(tuple, left_child_->GetOutputSchema()));
  }
  return key;
}

}  // namespace bustub
