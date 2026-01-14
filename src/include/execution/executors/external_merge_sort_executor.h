//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// external_merge_sort_executor.h
//
// Identification: src/include/execution/executors/external_merge_sort_executor.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "catalog/schema.h"
#include "common/config.h"
#include "common/macros.h"
#include "execution/execution_common.h"
#include "execution/executors/abstract_executor.h"
#include "execution/plans/sort_plan.h"
#include "storage/page/intermediate_result_page.h"
#include "storage/table/tuple.h"

namespace bustub {

class Iterator {
 public:
  typedef std::function<void(page_id_t)> ReleasePageCallback;

 public:
  explicit Iterator(std::vector<page_id_t> pages, BufferPoolManager *bpm, ReleasePageCallback release_page_callback);

  /**
   * Advance the iterator to the next tuple. If the current sort page is exhausted, move to the
   * next sort page.
   */
  auto operator++() -> Iterator &;

  /**
   * Dereference the iterator to get the current tuple in the sorted run that the iterator is
   * pointing to.
   */
  auto operator*() -> Tuple;

  auto End() -> bool { return cur_page_id_ == std::nullopt && pages_.empty(); }

 private:
  ReleasePageCallback release_page_callback_;
  BufferPoolManager *bpm_;

  /** The sorted run that the iterator is iterating on. */
  std::list<page_id_t> pages_;

  std::optional<page_id_t> cur_page_id_{std::nullopt};
  std::vector<Tuple> tuples_in_current_page_;
  size_t offset_{0};
};

/**
 * A data structure that holds the sorted tuples as a run during external merge sort.
 * Tuples might be stored in multiple pages, and tuples are ordered both within one page
 * and across pages.
 */
class MergeSortRun {
 public:
  typedef std::vector<page_id_t> PageIdVector;
  typedef std::function<bool(const Tuple &, const Tuple &)> Comparator;

  MergeSortRun(BufferPoolManager *bpm, Comparator &cmp_);

  auto Sort(PageIdVector &pages) -> PageIdVector;

 private:
  auto SortPage(page_id_t page_id) -> void;
  auto Merge(PageIdVector &left, PageIdVector &right) -> PageIdVector;

 private:
  /**
   * The buffer pool manager used to read sort pages. The buffer pool manager is responsible for
   * deleting the sort pages when they are no longer needed.
   */
  BufferPoolManager *bpm_;
  Comparator cmp_;
};

/**
 * ExternalMergeSortExecutor executes an external merge sort.
 *
 * In Spring 2025, only 2-way external merge sort is required.
 */
template <size_t K>
class ExternalMergeSortExecutor : public AbstractExecutor {
 public:
  ExternalMergeSortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                            std::unique_ptr<AbstractExecutor> &&child_executor);

  void Init() override;

  auto Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch, size_t batch_size)
      -> bool override;

  /** @return The output schema for the external merge sort */
  auto GetOutputSchema() const -> const Schema & override { return plan_->OutputSchema(); }

 private:
  auto LoadTupleIntoDiskPage() -> std::vector<page_id_t>;

 private:
  /** The sort plan node to be executed */
  const SortPlanNode *plan_;

  /** Compares tuples based on the order-bys */
  TupleComparator cmp_;

  std::unique_ptr<AbstractExecutor> child_executor_;

  std::unique_ptr<Iterator> itr_;
};

}  // namespace bustub
