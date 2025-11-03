//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// b_plus_tree_leaf_page.cpp
//
// Identification: src/storage/page/b_plus_tree_leaf_page.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstring>
#include <iterator>

#include "common/rid.h"
#include "storage/page/b_plus_tree_leaf_page.h"

namespace bustub {

/*****************************************************************************
 * HELPER METHODS AND UTILITIES
 *****************************************************************************/

/**
 * @brief Init method after creating a new leaf page
 *
 * After creating a new leaf page from buffer pool, must call initialize method to set default values,
 * including set page type, set current size to zero, set page id/parent id, set
 * next page id and set max size.
 *
 * @param max_size Max size of the leaf node
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::Init(int max_size) {
  SetPageType(IndexPageType::LEAF_PAGE);
  SetSize(0);
  SetMaxSize(max_size);

  SetNextPageId(INVALID_PAGE_ID);
  num_tombstones_ = 0;
}

/**
 * @brief Helper function for fetching tombstones of a page.
 * @return The last `NumTombs` keys with pending deletes in this page in order of recency (oldest at front).
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::GetTombstones() const -> std::vector<KeyType> {
  auto tombs = std::vector<KeyType>(num_tombstones_);
  for (size_t i = 0; i < num_tombstones_; ++i) {
    tombs[i] = key_array_[tombstones_[i]];
  }
  return std::move(tombs);
}

/**
 * Helper methods to set/get next page id
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::GetNextPageId() const -> page_id_t { return next_page_id_; }

FULL_INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::SetNextPageId(page_id_t next_page_id) { next_page_id_ = next_page_id; }

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Exist(const KeyType &key, const KeyComparator &comparator) const -> bool {
  return LookupIndex(key, comparator).has_value();
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Lookup(const KeyType &key, const KeyComparator &comparator) const
    -> std::optional<ValueType> {
  auto index = LookupIndex(key, comparator);
  if (!index.has_value()) {
    return std::nullopt;
  }
  return rid_array_[index.value()];
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::LookupIndex(const KeyType &key, const KeyComparator &comparator) const
    -> std::optional<size_t> {
  auto end = key_array_ + GetSize();
  auto wrapped_comparator = GenericComparatorWrapper(comparator);
  auto itr = std::lower_bound(key_array_, end, key, wrapped_comparator);
  if (itr == end || comparator(*itr, key) != 0) {
    return std::nullopt;
  }
  auto index = std::distance(key_array_, itr);
  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] == static_cast<size_t>(index)) {
      return std::nullopt;
    }
  }
  return index;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Split(page_id_t other_page_id, BPlusTreeLeafPage *other) -> void {
  int total_size = GetSize();
  int mid_index = total_size / 2;

  SetSize(mid_index);
  other->SetSize(total_size - mid_index);

  std::memcpy(other->key_array_, key_array_ + mid_index, (total_size - mid_index) * sizeof(KeyType));
  std::memcpy(other->rid_array_, rid_array_ + mid_index, (total_size - mid_index) * sizeof(ValueType));

  other->SetNextPageId(GetNextPageId());
  SetNextPageId(other_page_id);

  SplitTombstones(mid_index, other);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::SplitTombstones(size_t index, BPlusTreeLeafPage *other) -> void {
  size_t new_tombstones[LEAF_PAGE_TOMB_CNT];
  size_t new_num_tombs = 0;
  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] < static_cast<size_t>(index)) {
      new_tombstones[new_num_tombs++] = tombstones_[i];
    } else {
      other->tombstones_[other->num_tombstones_++] = tombstones_[i] - index;
    }
  }
  num_tombstones_ = new_num_tombs;
  std::copy_n(new_tombstones, new_num_tombs, other->tombstones_);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Insert(const KeyType &key, const ValueType &value, const KeyComparator &comparator)
    -> void {
  auto end = key_array_ + GetSize();
  auto wrapped_comparator = GenericComparatorWrapper(comparator);
  auto itr = std::upper_bound(key_array_, end, key, wrapped_comparator);

  size_t index = std::distance(key_array_, itr);
  if (itr != end && comparator(*itr, key) == 0) {
    Overwrite(value, index);
    return;
  }

  InsertInto(key, value, index);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Overwrite(const ValueType &value, size_t index) -> void {
  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] == index) {
      // remove tombstone
      std::memmove(&tombstones_[i], &tombstones_[i + 1], (num_tombstones_ - i - 1) * sizeof(size_t));
      --num_tombstones_;
      break;
    }
  }
  rid_array_[index] = value;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::InsertInto(const KeyType &key, const ValueType &value, size_t index) -> void {
  std::memmove(key_array_ + index + 1, key_array_ + index, (GetSize() - index) * sizeof(KeyType));
  std::memmove(rid_array_ + index + 1, rid_array_ + index, (GetSize() - index) * sizeof(ValueType));
  key_array_[index] = key;
  rid_array_[index] = value;

  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] >= index) {
      ++tombstones_[i];
    }
  }

  ChangeSizeBy(1);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Remove(size_t index) -> void {
  if (num_tombstones_ < LEAF_PAGE_TOMB_CNT) {
    tombstones_[num_tombstones_++] = index;
    return;
  }

  std::vector<size_t> to_remove{tombstones_, tombstones_ + num_tombstones_};
  to_remove.push_back(index);

  Clean(to_remove);

  ChangeSizeBy(-1 * (num_tombstones_ + 1));
  num_tombstones_ = 0;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Clean(std::vector<size_t> &to_remove) -> void {
  std::sort(to_remove.begin(), to_remove.end());
  to_remove.push_back(to_remove.size() + 1);

  size_t moved = to_remove[0];
  size_t prev = 0;

  for (size_t cur = 1; cur < to_remove.size(); ++cur) {
    auto n = to_remove[cur] - to_remove[prev] - 1;

    std::memmove(key_array_ + moved, key_array_ + to_remove[prev] + 1, n * sizeof(KeyType));
    std::memmove(rid_array_ + moved, rid_array_ + to_remove[prev] + 1, n * sizeof(ValueType));

    prev = cur;
    moved += n;
  }
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Lend(BPlusTreeLeafPage *right) -> KeyType {
  KeyType lend_key = key_array_[GetSize() - 1];
  ValueType lend_value = rid_array_[GetSize() - 1];

  right->InsertInto(lend_key, lend_value, 0);

  ChangeSizeBy(-1);
  return lend_key;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Merge(BPlusTreeLeafPage *right) -> void {
  std::memmove(key_array_ + GetSize(), right->key_array_, right->GetSize() * sizeof(KeyType));
  std::memmove(rid_array_ + GetSize(), right->rid_array_, right->GetSize() * sizeof(ValueType));

  ChangeSizeBy(right->GetSize());
  right->SetSize(0);
  right->num_tombstones_ = 0;
}

/*
 * Helper method to find and return the key associated with input "index" (a.k.a
 * array offset)
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::KeyAt(int index) const -> KeyType { return key_array_[index]; }

template class BPlusTreeLeafPage<GenericKey<4>, RID, GenericComparator<4>>;

template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>>;
template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 3>;
template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 2>;
template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 1>;
template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, -1>;

template class BPlusTreeLeafPage<GenericKey<16>, RID, GenericComparator<16>>;

template class BPlusTreeLeafPage<GenericKey<32>, RID, GenericComparator<32>>;

template class BPlusTreeLeafPage<GenericKey<64>, RID, GenericComparator<64>>;
}  // namespace bustub
