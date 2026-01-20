//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// window_function_executor.cpp
//
// Identification: src/execution/window_function_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/window_function_executor.h"
#include <unordered_map>
#include "common/config.h"
#include "common/macros.h"
#include "execution/execution_common.h"
#include "execution/plans/aggregation_plan.h"
#include "execution/plans/window_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

/**
 * Construct a new WindowFunctionExecutor instance.
 * @param exec_ctx The executor context
 * @param plan The window aggregation plan to be executed
 */
WindowFunctionExecutor::WindowFunctionExecutor(ExecutorContext *exec_ctx, const WindowFunctionPlanNode *plan,
                                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {}

/** Initialize the window aggregation */
void WindowFunctionExecutor::Init() {
  child_executor_->Init();

  results_.Reset();
  AggregateWindows();
}

/**
 * Yield the next tuple batch from the window aggregation.
 * @param[out] tuple_batch The next tuple batch produced by the window aggregation
 * @param[out] rid_batch The next tuple RID batch produced by the window aggregation
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 */
auto WindowFunctionExecutor::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                  size_t batch_size) -> bool {
  tuple_batch->clear();
  rid_batch->clear();

  for (size_t i = 0; i < batch_size; i++) {
    if (results_.Empty()) {
      break;
    }

    tuple_batch->push_back(*results_.Pop());
    rid_batch->push_back(RID{});
  }

  return !tuple_batch->empty();
}

auto WindowFunctionExecutor::AggregateWindows() -> void {
  auto child_tuples = LoadChildTuples();
  std::unordered_map<uint32_t, std::vector<Value>> window_func_values;

  for (auto [col_inx, func] : plan_->window_functions_) {
    auto values = AggregateWindow(child_tuples, func);
    window_func_values[col_inx] = std::move(values);
  }

  for (size_t i = 0; i < child_tuples.size(); i++) {
    std::vector<Value> values;

    for (size_t col_idx = 0; col_idx < plan_->columns_.size(); col_idx++) {
      if (plan_->window_functions_.find(col_idx) == plan_->window_functions_.end()) {
        auto expr = plan_->columns_[col_idx];
        values.push_back(expr->Evaluate(&child_tuples[i], child_executor_->GetOutputSchema()));
      } else {
        values.push_back(window_func_values[col_idx][i]);
      }
    }

    results_.Push({values, &plan_->OutputSchema()});
  }
}

auto WindowFunctionExecutor::LoadChildTuples() -> std::vector<Tuple> {
  std::vector<Tuple> total;

  std::vector<RID> rids;
  std::vector<Tuple> tuples;

  while (child_executor_->Next(&tuples, &rids, BUSTUB_BATCH_SIZE)) {
    total.insert(total.end(), tuples.begin(), tuples.end());

    tuples.clear();
    rids.clear();
  }

  // standard SQL does not guarantee order of rows without ORDER BY clause
  // But Bustub test cases may expect a stable order for window functions without ORDER BY
  // So we sort the tuples based on all columns to ensure a stable order
  SortTuplesIfWindowFuncHasOrderBy(total);

  return total;
}

auto WindowFunctionExecutor::SortTuplesIfWindowFuncHasOrderBy(std::vector<Tuple> &tuples) -> void {
  for (const auto &[_, func] : plan_->window_functions_) {
    if (!func.order_by_.empty()) {
      auto tuple_cmp = TupleComparator(func.order_by_, child_executor_->GetOutputSchema());
      std::sort(tuples.begin(), tuples.end(), tuple_cmp);
      return;
    }
  }
}

auto WindowFunctionExecutor::AggregateWindow(const std::vector<Tuple> &tuples,
                                             WindowFunctionPlanNode::WindowFunction &func) -> std::vector<Value> {
  auto schema = child_executor_->GetOutputSchema();

  std::vector<Value> results;
  results.resize(tuples.size());

  std::unordered_map<AggregateKey, std::vector<std::pair<size_t, Tuple>>> partitions;

  // partition the tuples
  for (size_t i = 0; i < tuples.size(); i++) {
    auto tuple = tuples[i];

    if (func.partition_by_.empty()) {
      // use a dummy key for no partition by
      partitions[AggregateKey{}].push_back({i, tuple});
      continue;
    }

    std::vector<Value> key_values;
    for (const auto &part_expr : func.partition_by_) {
      key_values.push_back(part_expr->Evaluate(&tuple, schema));
    }
    partitions[AggregateKey{key_values}].push_back({i, tuple});
  }

  // sort each partition if order by is specified
  if (!func.order_by_.empty()) {
    auto tuple_cmp = TupleComparator(func.order_by_, schema);
    auto cmp = [&](const std::pair<size_t, Tuple> &a, const std::pair<size_t, Tuple> &b) {
      return tuple_cmp(a.second, b.second);
    };
    for (auto &[_, partition] : partitions) {
      std::sort(partition.begin(), partition.end(), cmp);
    }
  }

  // set values
  for (auto &[_, partition] : partitions) {
    if (func.type_ == WindowFunctionType::Rank) {
      AggregateRankPartition(partition, results, func, schema);
    } else {
      AggregatePartition(partition, results, func, schema);
    }
  }

  return results;
}

auto WindowFunctionExecutor::AggregatePartition(const std::vector<std::pair<size_t, Tuple>> &partition,
                                                std::vector<Value> &results,
                                                WindowFunctionPlanNode::WindowFunction &func, Schema &schema) -> void {
  WindowFunctionValue window_func_value{func.type_};

  for (const auto &[idx, tuple] : partition) {
    window_func_value.Calculate(func.function_->Evaluate(&tuple, schema));

    if (!func.order_by_.empty()) {
      results[idx] = window_func_value.GetCurrentValue();
    }
  }

  if (func.order_by_.empty()) {
    for (const auto &[idx, _] : partition) {
      results[idx] = window_func_value.GetCurrentValue();
    }
  }
}

auto WindowFunctionExecutor::AggregateRankPartition(const std::vector<std::pair<size_t, Tuple>> &partition,
                                                    std::vector<Value> &results,
                                                    WindowFunctionPlanNode::WindowFunction &func, Schema &schema)
    -> void {
  RankValue rank_value;

  for (const auto &[idx, tuple] : partition) {
    rank_value.Calculate(GenerateSortKey(tuple, func.order_by_, schema));
    results[idx] = rank_value.GetCurrentValue();
  }
}

WindowFunctionValue::WindowFunctionValue(WindowFunctionType type) : type_(type) {
  switch (type_) {
    case WindowFunctionType::CountStarAggregate:
    case WindowFunctionType::CountAggregate:
      cur_ = ValueFactory::GetIntegerValue(0);
      break;
    case WindowFunctionType::SumAggregate:
    case WindowFunctionType::MinAggregate:
    case WindowFunctionType::MaxAggregate:
      cur_ = ValueFactory::GetNullValueByType(TypeId::INTEGER);
      break;
    default:
      UNIMPLEMENTED("unknown window function type");
  }
}

auto WindowFunctionValue::Calculate(const Value &value) -> void {
  BUSTUB_ENSURE(value.GetTypeId() == TypeId::INTEGER, "Value must be of type INTEGER");

  if (type_ == WindowFunctionType::CountStarAggregate) {
    // COUNT(*) counts all rows, so we increment for every non-null value
    cur_ = cur_.Add(ValueFactory::GetIntegerValue(1));
    return;
  }

  if (value.IsNull()) {
    return;
  }

  if (cur_.IsNull()) {
    BUSTUB_ENSURE(type_ != WindowFunctionType::CountAggregate, "CountAggregate should not have null current value");

    cur_ = value;
    return;
  }

  switch (type_) {
    case WindowFunctionType::CountAggregate:
      cur_ = cur_.Add(ValueFactory::GetIntegerValue(1));
      return;
    case WindowFunctionType::SumAggregate:
      cur_ = cur_.Add(value);
      return;
    case WindowFunctionType::MinAggregate:
      cur_ = cur_.Min(value);
      return;
    case WindowFunctionType::MaxAggregate:
      cur_ = cur_.Max(value);
      return;
    default:
      UNIMPLEMENTED("unknown window function type");
  }
}

auto RankValue::Calculate(const SortKey &values) -> void {
  rank_++;

  if (rank_ != 1) {
    BUSTUB_ENSURE(values.size() == last_value_.size(), "Values size must match last value size");

    if (IsSameSortKey(values, last_value_)) {
      return;
    }
  }

  last_value_ = values;
  cur_ = ValueFactory::GetIntegerValue(rank_);
}

}  // namespace bustub
