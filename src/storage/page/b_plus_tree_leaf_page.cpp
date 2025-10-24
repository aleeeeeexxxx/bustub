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
#include <sstream>

#include "common/exception.h"
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
  auto end = key_array_ + GetSize();
  auto wrapped_comparator = GenericComparatorWrapper(comparator);
  auto itr = std::lower_bound(key_array_, end, key, wrapped_comparator);
  if (itr == end || comparator(*itr, key) != 0) {
    return false;
  }
  auto index = std::distance(key_array_, itr);
  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] == static_cast<size_t>(index)) {
      return false;
    }
  }
  return true;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::IsFull() const -> bool { return GetSize() >= GetMaxSize(); }

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

  // move tombstones
  size_t new_tombstones_[LEAF_PAGE_TOMB_CNT];
  size_t new_num_tombs = 0;
  for (size_t i = 0; i < num_tombstones_; ++i) {
    if (tombstones_[i] >= static_cast<size_t>(mid_index)) {
      new_tombstones_[new_num_tombs++] = tombstones_[i] - mid_index;
    } else {
      other->tombstones_[other->num_tombstones_++] = tombstones_[i];
    }
  }
  num_tombstones_ = new_num_tombs;
  std::copy_n(new_tombstones_, new_num_tombs, other->tombstones_);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Insert(const KeyType &key, const ValueType &value, const KeyComparator &comparator)
    -> void {
  auto end = key_array_ + GetSize();
  auto wrapped_comparator = GenericComparatorWrapper(comparator);
  auto itr = std::lower_bound(key_array_, end, key, wrapped_comparator);

  size_t index = std::distance(key_array_, itr);
  if (itr != end && comparator(*itr, key) == 0) {
    // key exists, check tombstones
    for (size_t i = 0; i < num_tombstones_; ++i) {
      if (tombstones_[i] == index) {
        // remove tombstone
        std::memmove(&tombstones_[i], &tombstones_[i + 1], (num_tombstones_ - i - 1) * sizeof(size_t));
        --num_tombstones_;
        break;
      }
    }
    rid_array_[index] = value;
    return;
  }

  // shift right
  std::memmove(key_array_ + index + 1, key_array_ + index, (GetSize() - index) * sizeof(KeyType));
  std::memmove(rid_array_ + index + 1, rid_array_ + index, (GetSize() - index) * sizeof(ValueType));
  key_array_[index] = key;
  rid_array_[index] = value;

  ChangeSizeBy(1);
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
