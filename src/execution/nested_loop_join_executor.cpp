//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nested_loop_join_executor.cpp
//
// Identification: src/execution/nested_loop_join_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/nested_loop_join_executor.h"
#include "binder/table_ref/bound_join_ref.h"
#include "common/exception.h"
#include "common/macros.h"

namespace bustub {

auto CreateMergedTuple(const Tuple &left_tuple, const Schema &left_schema, const Tuple *right_tuple,
                       const Schema &right_schema, const Schema &output_schema) -> Tuple {
  std::vector<Value> values;

  for (size_t i = 0; i < left_schema.GetColumnCount(); i++) {
    values.push_back(left_tuple.GetValue(&left_schema, static_cast<uint32_t>(i)));
  }
  for (size_t i = 0; i < right_schema.GetColumnCount(); i++) {
    if (right_tuple) {
      values.push_back(right_tuple->GetValue(&right_schema, static_cast<uint32_t>(i)));
    } else {
      values.push_back(ValueFactory::GetNullValueByType(right_schema.GetColumn(i).GetType()));
    }
  }
  return Tuple(values, &output_schema);
}

/**
 * Construct a new NestedLoopJoinExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The nested loop join plan to be executed
 * @param left_executor The child executor that produces tuple for the left side of join
 * @param right_executor The child executor that produces tuple for the right side of join
 */
NestedLoopJoinExecutor::NestedLoopJoinExecutor(ExecutorContext *exec_ctx, const NestedLoopJoinPlanNode *plan,
                                               std::unique_ptr<AbstractExecutor> &&left_executor,
                                               std::unique_ptr<AbstractExecutor> &&right_executor)
    : AbstractExecutor(exec_ctx),
      plan_(plan),
      left_executor_(std::move(left_executor)),
      right_executor_(std::move(right_executor)) {
  if (plan->GetJoinType() != JoinType::LEFT && plan->GetJoinType() != JoinType::INNER) {
    // Note for Spring 2025: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
}

/** Initialize the join */
void NestedLoopJoinExecutor::Init() {
  left_executor_->Init();
  right_executor_->Init();

  std::vector<RID> rid_batch;
  std::vector<Tuple> tuple_batch;
  while (right_executor_->Next(&tuple_batch, &rid_batch, BUSTUB_BATCH_SIZE)) {
    right_tuples_.insert(right_tuples_.end(), tuple_batch.begin(), tuple_batch.end());

    tuple_batch.clear();
    rid_batch.clear();
  }
}

/**
 * Yield the next tuple batch from the join.
 * @param[out] tuple_batch The next tuple batch produced by the join
 * @param[out] rid_batch The next tuple RID batch produced by the join
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 */
auto NestedLoopJoinExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                  size_t batch_size) -> bool {
  if (cur_left_tuple_ == nullptr && !LoadNextLeftTuple()) {
    return false;
  }

  while (tuple_batch->size() < batch_size) {
    if (cur_right_idx_ >= right_tuples_.size()) {
      if (plan_->GetJoinType() == JoinType::LEFT && !joined_) {
        CreateMergedTuple(*cur_left_tuple_, left_executor_->GetOutputSchema(), nullptr,
                          right_executor_->GetOutputSchema(), plan_->OutputSchema());
      }
      if (!LoadNextLeftTuple()) {
        break;
      }

      cur_right_idx_ = 0;
      joined_ = false;
      continue;
    }

    auto right_tuple = right_tuples_[cur_right_idx_++];

    if (plan_->predicate_
            ->EvaluateJoin(cur_left_tuple_.get(), left_executor_->GetOutputSchema(), &right_tuple,
                           right_executor_->GetOutputSchema())
            .GetAs<bool>()) {
      joined_ = true;
      CreateMergedTuple(*cur_left_tuple_, left_executor_->GetOutputSchema(), &right_tuple,
                        right_executor_->GetOutputSchema(), plan_->OutputSchema());
    }
  }

  return tuple_batch->size() > 0;
}

auto NestedLoopJoinExecutor::LoadNextLeftTuple() -> bool {
  cur_left_tuple_ = nullptr;

  std::vector<RID> rid_batch;
  std::vector<Tuple> tuple_batch;

  auto has_next = left_executor_->Next(&tuple_batch, &rid_batch, 1);
  if (!has_next) {
    return false;
  }

  cur_left_tuple_ = std::make_unique<Tuple>(tuple_batch[0]);
  return true;
}

}  // namespace bustub
