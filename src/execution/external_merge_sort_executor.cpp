//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// external_merge_sort_executor.cpp
//
// Identification: src/execution/external_merge_sort_executor.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/external_merge_sort_executor.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>
#include "common/config.h"
#include "common/macros.h"
#include "execution/execution_common.h"
#include "execution/executors/abstract_executor.h"
#include "execution/plans/sort_plan.h"
#include "storage/page/page_guard.h"
#include "storage/table/tuple.h"

namespace bustub {

Iterator::Iterator(std::vector<page_id_t> pages, BufferPoolManager *bpm,
                   ReleasePageCallback release_page_callback = nullptr)
    : release_page_callback_(release_page_callback), bpm_(bpm), pages_(pages.begin(), pages.end()) {}

auto Iterator::operator++() -> Iterator & {
  BUSTUB_ENSURE(!End(), "Iterator has reached the end");

  if (++offset_ < tuples_in_current_page_.size()) {
    return *this;
  }

  cur_page_id_ = std::nullopt;
  tuples_in_current_page_.clear();
  offset_ = 0;

  if (pages_.empty()) {
    return *this;
  }

  cur_page_id_ = pages_.front();
  pages_.pop_front();

  auto guard = bpm_->ReadPage(cur_page_id_.value());
  auto page = guard.As<IntermediateResultPage>();

  page->ToTuples(tuples_in_current_page_);
  BUSTUB_ENSURE(tuples_in_current_page_.size() > 0, "Page should contain at least one tuple");

  if (release_page_callback_) {
    release_page_callback_(cur_page_id_.value());
  }
  return *this;
}

auto Iterator::operator*() -> Tuple {
  BUSTUB_ENSURE(!End(), "Iterator has reached the end");

  if (cur_page_id_ == std::nullopt) {
    ++(*this);
  }

  BUSTUB_ENSURE(offset_ < tuples_in_current_page_.size(), "Iterator out of bounds");
  return tuples_in_current_page_[offset_];
}

MergeSortRun::MergeSortRun(BufferPoolManager *bpm, Comparator &cmp) : bpm_(bpm), cmp_(cmp) {}

auto MergeSortRun::Sort(const PageIdVector &pages) -> PageIdVector {
  BUSTUB_ENSURE(!pages.empty(), "Pages vector should not be empty");

  if (pages.size() == 1) {
    SortPage(pages[0]);
    return pages;
  }

  PageIdVector left, right;
  for (size_t i = 0; i < pages.size(); i++) {
    if (i < pages.size() / 2) {
      left.push_back(pages[i]);
    } else {
      right.push_back(pages[i]);
    }
  }

  auto sorted_left = Sort(left);
  auto sorted_right = Sort(right);

  return Merge(sorted_left, sorted_right);
}

auto MergeSortRun::Merge(PageIdVector &left, PageIdVector &right) -> PageIdVector {
  PageIdVector result_pages;
  std::optional<WritePageGuard> guard;
  IntermediateResultPage *page;

  std::list<page_id_t> free_list;
  Iterator::ReleasePageCallback callback = [&](page_id_t page_id) { free_list.push_back(page_id); };

  Iterator left_itr{left, bpm_, callback};
  Iterator right_itr{right, bpm_, callback};

  while (!left_itr.End() || !right_itr.End()) {
    Iterator *to_insert_itr = &right_itr;
    if (right_itr.End() || (!left_itr.End() && cmp_(*left_itr, *right_itr))) {
      to_insert_itr = &left_itr;
    }

    Tuple to_insert = **to_insert_itr;

    if (!guard.has_value()) {
      page_id_t page_id;
      if (free_list.empty()) {
        page_id = bpm_->NewPage();
      } else {
        page_id = free_list.front();
        free_list.pop_front();
      }

      result_pages.push_back(page_id);
      guard = bpm_->WritePage(page_id);

      page = guard->AsMut<IntermediateResultPage>();
      page->Reset();
    }

    if (page->CanInsert(to_insert)) {
      page->InsertTuple(to_insert);
      ++(*to_insert_itr);
    } else {
      guard->Drop();
      guard = std::nullopt;
    }
  }

  return result_pages;
}

auto MergeSortRun::SortPage(page_id_t page_id) -> void {
  std::vector<Tuple> buffer;
  auto guard = bpm_->WritePage(page_id);
  auto page = guard.AsMut<IntermediateResultPage>();

  page->ToTuples(buffer);
  page->Reset();

  std::sort(buffer.begin(), buffer.end(), cmp_);

  for (const auto &tuple : buffer) {
    BUSTUB_ENSURE(page->CanInsert(tuple), "Sort should not change the original data size");
    page->InsertTuple(tuple);
  }
}

template <size_t K>
ExternalMergeSortExecutor<K>::ExternalMergeSortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                                                        std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), cmp_(plan->GetOrderBy()), child_executor_(std::move(child_executor)) {}

/** Initialize the external merge sort */
template <size_t K>
void ExternalMergeSortExecutor<K>::Init() {
  child_executor_->Init();

  auto page_ids = LoadTupleIntoDiskPage();

  // for (auto page_id : page_ids) {
  //   std::cout << "loaded page id: " << page_id << std::endl;

  //   auto bpm = exec_ctx_->GetBufferPoolManager();
  //   auto guard = bpm->WritePage(page_id);
  //   auto page = reinterpret_cast<IntermediateResultPage *>(guard.GetDataMut());

  //   page->PrintTuples(child_executor_->GetOutputSchema());
  // }

  if (!page_ids.empty()) {
    auto compare = GetTupleComparator(plan_->GetOrderBy(), child_executor_->GetOutputSchema(), cmp_);
    MergeSortRun merge_sort_run(exec_ctx_->GetBufferPoolManager(), compare);

    page_ids = merge_sort_run.Sort(page_ids);
  }

  std::cout << "ExternalMergeSortExecutor: sorted " << page_ids.size() << " pages." << std::endl;

  itr_ = std::make_unique<Iterator>(page_ids, exec_ctx_->GetBufferPoolManager());
}

template <size_t K>
auto ExternalMergeSortExecutor<K>::LoadTupleIntoDiskPage() -> std::vector<page_id_t> {
  std::vector<page_id_t> page_ids;

  std::vector<RID> rids;

  WritePageGuard guard;
  IntermediateResultPage *page;

  ReusableCache tuples;

  std::optional<page_id_t> cur = std::nullopt;

  while (true) {
    if (tuples.Empty()) {
      tuples.Reset();
      rids.clear();

      if (!child_executor_->Next(tuples.Raw(), &rids, BUSTUB_BATCH_SIZE)) {
        break;
      }
      // start another round in case no tuples were loaded
      continue;
    }

    if (!cur.has_value()) {
      auto new_page_id = exec_ctx_->GetBufferPoolManager()->NewPage();
      page_ids.push_back(new_page_id);
      cur = new_page_id;

      std::cout << "ExternalMergeSortExecutor: created new page " << new_page_id << std::endl;

      guard = exec_ctx_->GetBufferPoolManager()->WritePage(new_page_id);
      page = guard.AsMut<IntermediateResultPage>();
      page->Reset();
    }

    auto tuple = tuples.Peek();

    if (page->CanInsert(tuple)) {
      page->InsertTuple(tuple);
      tuples.Next();

      // std::cout << "ExternalMergeSortExecutor: inserted tuple into page " << cur.value() << " "
      //           << tuple.ToString(&child_executor_->GetOutputSchema()) << std::endl;
    } else {
      guard.Drop();
      cur = std::nullopt;
    }
  }

  return page_ids;
}

/**
 * Yield the next tuple batch from the external merge sort.
 * @param[out] tuple_batch The next tuple batch produced by the external merge sort.
 * @param[out] rid_batch The next tuple RID batch produced by the external merge sort.
 * @param batch_size The number of tuples to be included in the batch (default: BUSTUB_BATCH_SIZE)
 * @return `true` if a tuple was produced, `false` if there are no more tuples
 */
template <size_t K>
auto ExternalMergeSortExecutor<K>::Next(std::vector<bustub::Tuple> *tuple_batch, std::vector<bustub::RID> *rid_batch,
                                        size_t batch_size) -> bool {
  tuple_batch->clear();
  rid_batch->clear();

  BUSTUB_ENSURE(itr_ != nullptr, "itr should be initialized.");

  for (size_t i = 0; i < batch_size; i++) {
    if (itr_->End()) {
      break;
    }

    tuple_batch->push_back(**itr_);
    rid_batch->push_back(RID{});
    ++(*itr_);
  }

  return !tuple_batch->empty();
}

template class ExternalMergeSortExecutor<2>;

}  // namespace bustub
