//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// b_plus_tree_internal_page.cpp
//
// Identification: src/storage/page/b_plus_tree_internal_page.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/page/b_plus_tree_internal_page.h"
#include <algorithm>

namespace bustub {
/*****************************************************************************
 * HELPER METHODS AND UTILITIES
 *****************************************************************************/

const int MIN_INTERNAL_PAGE_SIZE = 3;

/**
 * @brief Init method after creating a new internal page.
 *
 * Writes the necessary header information to a newly created page,
 * including set page type, set current size, set page id, set parent id and set max page size,
 * must be called after the creation of a new page to make a valid BPlusTreeInternalPage.
 *
 * @param max_size Maximal size of the page
 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_INTERNAL_PAGE_TYPE::Init(int max_size) {
  BUSTUB_ENSURE(max_size >= MIN_INTERNAL_PAGE_SIZE, "Invalid internal page size");

  SetPageType(IndexPageType::INTERNAL_PAGE);
  SetSize(0);
  SetMaxSize(max_size);
}

/**
 * @brief Helper method to get/set the key associated with input "index"(a.k.a
 * array offset).
 *
 * @param index The index of the key to get. Index must be non-zero.
 * @return Key at index
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::KeyAt(int index) const -> KeyType { return key_array_[index]; }

/**
 * @brief Set key at the specified index.
 *
 * @param index The index of the key to set. Index must be non-zero.
 * @param key The new value for key
 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_INTERNAL_PAGE_TYPE::SetKeyAt(int index, const KeyType &key) { key_array_[index] = key; }

/**
 * @brief Helper method to get the value associated with input "index"(a.k.a array
 * offset)
 *
 * @param index The index of the value to get.
 * @return Value at index
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::ValueAt(int index) const -> ValueType { return page_id_array_[index]; }

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Search(const KeyType &key, const KeyComparator &comparator) const -> ValueType {
  auto index = GetTargetPageIndex(key, comparator);
  return ValueAt(index);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::GetTargetPageIndex(const KeyType &key, const KeyComparator &comparator) const
    -> size_t {
  return UpperBound(key, comparator) - 1;
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::UpperBound(const KeyType &key, const KeyComparator &comparator) const -> int {
  auto start = key_array_ + 1;
  auto end = key_array_ + GetSize();
  auto wrapped_comparator = GenericComparatorWrapper(comparator);
  auto itr = std::upper_bound(start, end, key, wrapped_comparator);
  return static_cast<int>(std::distance(key_array_, itr));
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::SplitAndInsert(BPlusTreeInternalPage *other, const KeyType &key,
                                                    const ValueType &value, const KeyComparator &comparator)
    -> KeyType {
  int total_size = GetSize();
  int mid_index = total_size / 2;
  bool insert_into_other = false;

  if (comparator(key, key_array_[mid_index]) > 0) {
    mid_index++;
    insert_into_other = true;
  }

  SetSize(mid_index);
  other->SetSize(total_size - mid_index);

  std::memcpy(other->key_array_, key_array_ + mid_index, (total_size - mid_index) * sizeof(KeyType));
  std::memcpy(other->page_id_array_, page_id_array_ + mid_index, (total_size - mid_index + 1) * sizeof(ValueType));

  if (insert_into_other) {
    if (comparator(key, other->KeyAt(0)) < 0) {
      other->InsertInto(0, key, value);
    } else {
      other->Insert(key, value, comparator);
    }
  } else {
    Insert(key, value, comparator);
  }

  return other->KeyAt(0);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Insert(const KeyType &key, const ValueType &value, const KeyComparator &comparator)
    -> void {
  auto index = UpperBound(key, comparator);
  InsertInto(index, key, value);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::InsertInto(size_t index, const KeyType &key, const ValueType &value) -> void {
  std::memmove(key_array_ + index + 1, key_array_ + index, (GetSize() - index) * sizeof(KeyType));
  std::memmove(page_id_array_ + index + 1, page_id_array_ + index, (GetSize() - index) * sizeof(ValueType));

  key_array_[index] = key;
  page_id_array_[index] = value;

  ChangeSizeBy(1);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Init(const KeyType &key, const ValueType &value1, const ValueType &value2)
    -> void {
  key_array_[1] = key;
  page_id_array_[0] = value1;
  page_id_array_[1] = value2;
  SetSize(2);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::SearchCurrentAndSibling(const KeyType &key, CurAndSibling &result,
                                                             const KeyComparator &comparator) const -> void {
  auto index = GetTargetPageIndex(key, comparator);
  result.cur_ = ValueAt(index);
  result.cur_index_ = index;

  auto sibling = static_cast<int>(index) - 1;
  if (sibling >= 0) {
    result.is_left_ = true;
  } else {
    sibling = index + 1;
    result.is_left_ = false;
  }

  result.sibling_ = ValueAt(sibling);
  result.sibling_index_ = sibling;
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::LendToRight(BPlusTreeInternalPage *right) -> KeyType {
  KeyType lend_key = key_array_[GetSize() - 1];
  ValueType lend_value = page_id_array_[GetSize() - 1];

  right->InsertInto(0, lend_key, lend_value);

  ChangeSizeBy(-1);
  return lend_key;
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::LendToLeft(BPlusTreeInternalPage *left) -> KeyType {
  KeyType lend_key = key_array_[0];
  ValueType lend_value = page_id_array_[0];

  left->InsertInto(left->GetSize(), lend_key, lend_value);

  Remove(0);
  return lend_key;
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Merge(BPlusTreeInternalPage *right) -> void {
  std::memmove(key_array_ + GetSize(), right->key_array_, right->GetSize() * sizeof(KeyType));
  std::memmove(page_id_array_ + GetSize(), right->page_id_array_, right->GetSize() * sizeof(ValueType));

  ChangeSizeBy(right->GetSize());
  right->SetSize(0);
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Remove(size_t index) -> void {
  std::memmove(key_array_ + index, key_array_ + index + 1, (GetSize() - index - 1) * sizeof(KeyType));
  std::memmove(page_id_array_ + index, page_id_array_ + index + 1, (GetSize() - index - 1) * sizeof(ValueType));

  ChangeSizeBy(-1);
}

// valuetype for internalNode should be page id_t
template class BPlusTreeInternalPage<GenericKey<4>, page_id_t, GenericComparator<4>>;
template class BPlusTreeInternalPage<GenericKey<8>, page_id_t, GenericComparator<8>>;
template class BPlusTreeInternalPage<GenericKey<16>, page_id_t, GenericComparator<16>>;
template class BPlusTreeInternalPage<GenericKey<32>, page_id_t, GenericComparator<32>>;
template class BPlusTreeInternalPage<GenericKey<64>, page_id_t, GenericComparator<64>>;
}  // namespace bustub
