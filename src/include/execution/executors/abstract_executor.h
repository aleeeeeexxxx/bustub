//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// abstract_executor.h
//
// Identification: src/include/execution/executors/abstract_executor.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

#include "execution/executor_context.h"
#include "storage/table/tuple.h"

#include "type/value_factory.h"

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

class ExecutorContext;
/**
 * The AbstractExecutor implements the Volcano tuple-at-a-time iterator model.
 * This is the base class from which all executors in the BustTub execution
 * engine inherit, and defines the minimal interface that all executors support.
 */
class AbstractExecutor {
 public:
  /**
   * Construct a new AbstractExecutor instance.
   * @param exec_ctx the executor context that the executor runs with
   */
  explicit AbstractExecutor(ExecutorContext *exec_ctx) : exec_ctx_{exec_ctx} {}

  /** Virtual destructor. */
  virtual ~AbstractExecutor() = default;

  /**
   * Initialize the executor.
   * @warning This function must be called before Next() is called!
   */
  virtual void Init() = 0;

  /**
   * Yield the next tuple from this executor.
   * @param[out] tuple The next tuple produced by this executor
   * @param[out] rid The next tuple RID produced by this executor
   * @return `true` if a tuple was produced, `false` if there are no more tuples
   */
  virtual auto Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch, size_t batch_size)
      -> bool = 0;

  /** @return The schema of the tuples that this executor produces */
  virtual auto GetOutputSchema() const -> const Schema & = 0;

  /** @return The executor context in which this executor runs */
  auto GetExecutorContext() -> ExecutorContext * { return exec_ctx_; }

  auto GenerateResultTuple(size_t value) -> Tuple {
    std::vector<Value> values;
    values.emplace_back(ValueFactory::GetIntegerValue(static_cast<int32_t>(value)));
    return Tuple(values, &GetOutputSchema());
  }

 protected:
  /** The executor context in which the executor runs */
  ExecutorContext *exec_ctx_;
};

}  // namespace bustub
