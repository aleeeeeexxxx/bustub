//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nested_loop_join_executor.h
//
// Identification: src/include/execution/executors/nested_loop_join_executor.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include "execution/executor_context.h"
#include "execution/executors/abstract_executor.h"
#include "execution/plans/nested_loop_join_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

class ReusableCache {
 public:
  ReusableCache() = default;

  auto Raw() -> std::vector<Tuple> * { return &cache_; }
  auto Reset() -> void {
    next_ = 0;
    cache_.clear();
  }
  auto Empty() -> bool { return next_ >= cache_.size(); }
  Tuple *Pop() {
    BUSTUB_ASSERT(!Empty(), "Cache is empty");
    return &cache_[next_++];
  }

 private:
  std::vector<Tuple> cache_;
  size_t next_ = 0;
};

/**
 * NestedLoopJoinExecutor executes a nested-loop JOIN on two tables.
 */
class NestedLoopJoinExecutor : public AbstractExecutor {
 public:
  NestedLoopJoinExecutor(ExecutorContext *exec_ctx, const NestedLoopJoinPlanNode *plan,
                         std::unique_ptr<AbstractExecutor> &&left_executor,
                         std::unique_ptr<AbstractExecutor> &&right_executor);

  void Init() override;

  auto Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch, size_t batch_size)
      -> bool override;

  /** @return The output schema for the insert */
  auto GetOutputSchema() const -> const Schema & override { return plan_->OutputSchema(); };

 private:
  /** The NestedLoopJoin plan node to be executed. */
  const NestedLoopJoinPlanNode *plan_;

  std::unique_ptr<AbstractExecutor> left_executor_;
  std::unique_ptr<AbstractExecutor> right_executor_;

  std::unique_ptr<Tuple> cur_left_tuple_;
  bool joined_ = false;
  ReusableCache right_cache_;

 private:
  auto LoadNextLeftTuple() -> bool;
};

}  // namespace bustub
