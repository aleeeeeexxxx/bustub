//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// index_iterator.cpp
//
// Identification: src/storage/index/index_iterator.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

/**
 * index_iterator.cpp
 */
#include <algorithm>
#include <cassert>
#include "buffer/arc_replacer.h"
#include "common/macros.h"

#include "storage/index/index_iterator.h"

namespace bustub {

/**
 * @note you can change the destructor/constructor method here
 * set your own input parameters
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::IndexIterator(page_id_t page_id, size_t index, PageGuard &&guard,
                                  std::shared_ptr<TracedBufferPoolManager> bpm, KeyComparator comparator)
    : page_id_(page_id),
      index_(index),
      guard_(std::move(guard)),
      end_(false),
      comparator_(comparator),
      bpm_(std::move(bpm)) {}

FULL_INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::IndexIterator(KeyComparator comparator) : end_(true), comparator_(comparator) {}

FULL_INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::IndexIterator(IndexIterator &&that) noexcept : comparator_(that.comparator_) {
  Move(std::move(that));
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator=(IndexIterator &&that) noexcept -> IndexIterator & {
  if (this != &that) {
    comparator_ = that.comparator_;
    Move(std::move(that));
  }
  return *this;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
void INDEXITERATOR_TYPE::Move(IndexIterator &&that) {
  page_id_ = that.page_id_;
  index_ = that.index_;
  guard_ = std::move(that.guard_);
  end_ = that.end_;
  bpm_ = std::move(that.bpm_);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::~IndexIterator(){};  // NOLINT

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::IsEnd() -> bool { return end_; }

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator*() -> std::pair<const KeyType &, const ValueType &> {
  auto page = guard_.As<LeafPage>();
  cur_pair_.first = page->KeyAt(index_);
  cur_pair_.second = page->ValueAt(index_);
  return {cur_pair_.first, cur_pair_.second};
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator++() -> INDEXITERATOR_TYPE & {
  if (end_) {
    return *this;
  }

  while (true) {
    auto page = guard_.As<LeafPage>();
    auto next_index = page->Next(index_);
    if (next_index.has_value()) {
      index_ = next_index.value();
      break;
    } else {
      page_id_t next_page_id = page->GetNextPageId();

      if (next_page_id == INVALID_PAGE_ID) {
        end_ = true;
        break;
      } else {
        guard_ = std::move(bpm_->ReadPage(next_page_id, AccessType::Scan));
        index_ = -1;
      }
    }
  }

  return *this;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator==(const IndexIterator &itr) const -> bool {
  if (end_ && itr.end_) {
    return true;
  }

  return page_id_ == itr.page_id_ && index_ == itr.index_;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator!=(const IndexIterator &itr) const -> bool { return !(*this == itr); }

template class IndexIterator<GenericKey<4>, RID, GenericComparator<4>>;

template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>>;
template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>, 3>;
template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>, 2>;
template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>, 1>;
template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>, -1>;

template class IndexIterator<GenericKey<16>, RID, GenericComparator<16>>;

template class IndexIterator<GenericKey<32>, RID, GenericComparator<32>>;

template class IndexIterator<GenericKey<64>, RID, GenericComparator<64>>;

}  // namespace bustub
