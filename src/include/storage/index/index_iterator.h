//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// index_iterator.h
//
// Identification: src/include/storage/index/index_iterator.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

/**
 * index_iterator.h
 * For range scan of b+ tree
 */
#pragma once
#include <utility>
#include "buffer/traced_buffer_pool_manager.h"
#include "common/config.h"
#include "common/macros.h"
#include "storage/page/b_plus_tree_internal_page.h"
#include "storage/page/b_plus_tree_leaf_page.h"
#include "storage/page/page_guard.h"

namespace bustub {

#define INDEXITERATOR_TYPE IndexIterator<KeyType, ValueType, KeyComparator, NumTombs>
#define SHORT_INDEXITERATOR_TYPE IndexIterator<KeyType, ValueType, KeyComparator>

FULL_INDEX_TEMPLATE_ARGUMENTS_DEFN
class IndexIterator {
  using InternalPage = BPlusTreeInternalPage<KeyType, page_id_t, KeyComparator>;
  using LeafPage = BPlusTreeLeafPage<KeyType, ValueType, KeyComparator, NumTombs>;

 public:
  // you may define your own constructor based on your member variables
  IndexIterator(KeyComparator comparator);
  IndexIterator(page_id_t page_id, size_t index, PageGuard &&guard, std::shared_ptr<TracedBufferPoolManager> bpm,
                KeyComparator comparator);

  IndexIterator(IndexIterator &) = delete;
  auto operator=(const IndexIterator &) -> IndexIterator & = delete;

  IndexIterator(IndexIterator &&that) noexcept;
  auto operator=(IndexIterator &&that) noexcept -> IndexIterator &;

  ~IndexIterator();  // NOLINT

  auto IsEnd() -> bool;

  auto operator*() -> std::pair<const KeyType &, const ValueType &>;

  auto operator++() -> IndexIterator &;

  auto operator==(const IndexIterator &itr) const -> bool { UNIMPLEMENTED("TODO(P2): Add implementation."); }

  auto operator!=(const IndexIterator &itr) const -> bool { UNIMPLEMENTED("TODO(P2): Add implementation."); }

 private:
  void move(IndexIterator &&that);

 private:
  page_id_t page_id_;
  size_t index_;
  PageGuard guard_;
  bool end_;
  KeyComparator comparator_;
  std::shared_ptr<TracedBufferPoolManager> bpm_;

 private:
  mutable std::pair<KeyType, ValueType> cur_pair_;
};

}  // namespace bustub
