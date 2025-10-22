//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// b_plus_tree.cpp
//
// Identification: src/storage/index/b_plus_tree.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/index/b_plus_tree.h"
#include "buffer/traced_buffer_pool_manager.h"
#include "common/config.h"
#include "storage/index/b_plus_tree_debug.h"

namespace bustub {

FULL_INDEX_TEMPLATE_ARGUMENTS
BPLUSTREE_TYPE::BPlusTree(std::string name, page_id_t header_page_id, BufferPoolManager *buffer_pool_manager,
                          const KeyComparator &comparator, int leaf_max_size, int internal_max_size)
    : bpm_(std::make_shared<TracedBufferPoolManager>(buffer_pool_manager)),
      index_name_(std::move(name)),
      comparator_(std::move(comparator)),
      leaf_max_size_(leaf_max_size),
      internal_max_size_(internal_max_size),
      header_page_id_(header_page_id) {
  WritePageGuard guard = bpm_->WritePage(header_page_id_);
  auto root_page = guard.AsMut<BPlusTreeHeaderPage>();
  root_page->root_page_id_ = INVALID_PAGE_ID;
}

/**
 * @brief Helper function to decide whether current b+tree is empty
 * @return Returns true if this B+ tree has no keys and values.
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::IsEmpty() const -> bool { UNIMPLEMENTED("TODO(P2): Add implementation."); }

/*****************************************************************************
 * SEARCH
 *****************************************************************************/
/**
 * @brief Return the only value that associated with input key
 *
 * This method is used for point query
 *
 * @param key input key
 * @param[out] result vector that stores the only value that associated with input key, if the value exists
 * @return : true means key exists
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetValue(const KeyType &key, std::vector<ValueType> *result) -> bool {
  UNIMPLEMENTED("TODO(P2): Add implementation.");
  // Declaration of context instance. Using the Context is not necessary but advised.
  Context ctx;
}

/*****************************************************************************
 * INSERTION
 *****************************************************************************/
/**
 * @brief Insert constant key & value pair into b+ tree
 *
 * if current tree is empty, start new tree, update root page id and insert
 * entry; otherwise, insert into leaf page.
 *
 * @param key the key to insert
 * @param value the value associated with key
 * @return: since we only support unique key, if user try to insert duplicate
 * keys return false; otherwise, return true.
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(const KeyType &key, const ValueType &value) -> bool {
  InsertResult result;
  Insert(key, value, GetRootPageId(), result);
  return result.success;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(const KeyType &key, const ValueType &value, page_id_t page_id,
                            InsertResult &result) -> void {
  auto page_guard = bpm_->WritePage(page_id);
  auto page = page_guard.AsMut<BPlusTreePage>();

  if (page->IsLeafPage()) {
    auto leaf_page = page_guard.AsMut<LeafPage>();
    InsertIntoLeafPage(key, value, leaf_page, result);
  } else {
    auto internal_page = page_guard.AsMut<InternalPage>();
    Insert(key, value, internal_page->Search(key, comparator_), result);

    if (result.split_page_id != INVALID_PAGE_ID) {
      InsertIntoPage(result.start_key, result.split_page_id, internal_page, result);
    }
  }

  if (page_id == header_page_id_ && result.split_page_id != INVALID_PAGE_ID) {
    // Root page split
    auto [new_page_id, new_internal_page] = CreateNewInternalPage();
    new_internal_page->Insert(result.start_key, result.split_page_id, comparator_);
    new_internal_page->Insert(result.start_key, header_page_id_, comparator_);

    header_page_id_ = new_page_id;
  }
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::InsertIntoPage(const KeyType &key, page_id_t page_id, InternalPage *page,
                                    InsertResult &result) -> void {
  if (!page->IsFull()) {
    page->Insert(key, page_id, comparator_);
    result.split_page_id = INVALID_PAGE_ID;
    return;
  }

  auto [new_page_id, new_internal_page] = CreateNewInternalPage();
  auto mid = page->Split(new_internal_page);

  if (comparator_(key, mid) < 0) {
    page->Insert(key, page_id, comparator_);
  } else {
    new_internal_page->Insert(key, page_id, comparator_);
  }

  result.split_page_id = new_page_id;
  result.start_key = mid;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::InsertIntoLeafPage(const KeyType &key, const ValueType &value, LeafPage *page,
                                        InsertResult &result) -> void {
  if (page->Exist(key, comparator_)) {
    result.success = false;
    return;
  }

  result.success = true;

  if (!page->IsFull()) {
    page->Insert(key, value, comparator_);
    return;
  }

  auto [new_page_id, new_leaf_page] = CreateNewLeafPage();
  auto mid = page->Split(new_leaf_page);

  if (comparator_(key, mid) < 0) {
    page->Insert(key, value, comparator_);
  } else {
    new_leaf_page->Insert(key, value, comparator_);
  }

  result.start_key = mid;
  result.split_page_id = new_page_id;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::CreateNewInternalPage() -> std::pair<page_id_t, InternalPage *> {
  auto new_page_id = bpm_->NewPage();
  auto guard = bpm_->WritePage(new_page_id, AccessType::Index);

  auto new_internal_page = guard.AsMut<InternalPage>();
  new_internal_page->Init(internal_max_size_);

  return {new_page_id, new_internal_page};
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::CreateNewLeafPage() -> std::pair<page_id_t, LeafPage *> {
  auto new_page_id = bpm_->NewPage();
  auto guard = bpm_->WritePage(new_page_id, AccessType::Index);

  auto new_leaf_page = guard.AsMut<LeafPage>();
  new_leaf_page->Init(leaf_max_size_);

  return {new_page_id, new_leaf_page};
}

/*****************************************************************************
 * REMOVE
 *****************************************************************************/
/**
 * @brief Delete key & value pair associated with input key
 * If current tree is empty, return immediately.
 * If not, User needs to first find the right leaf page as deletion target, then
 * delete entry from leaf page. Remember to deal with redistribute or merge if
 * necessary.
 *
 * @param key input key
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
void BPLUSTREE_TYPE::Remove(const KeyType &key) {
  // Declaration of context instance.
  Context ctx;
  UNIMPLEMENTED("TODO(P2): Add implementation.");
}

/*****************************************************************************
 * INDEX ITERATOR
 *****************************************************************************/
/**
 * @brief Input parameter is void, find the leftmost leaf page first, then construct
 * index iterator
 *
 * You may want to implement this while implementing Task #3.
 *
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin() -> INDEXITERATOR_TYPE { UNIMPLEMENTED("TODO(P2): Add implementation."); }

/**
 * @brief Input parameter is low key, find the leaf page that contains the input key
 * first, then construct index iterator
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin(const KeyType &key) -> INDEXITERATOR_TYPE { UNIMPLEMENTED("TODO(P2): Add implementation."); }

/**
 * @brief Input parameter is void, construct an index iterator representing the end
 * of the key/value pair in the leaf node
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::End() -> INDEXITERATOR_TYPE { UNIMPLEMENTED("TODO(P2): Add implementation."); }

/**
 * @return Page id of the root of this tree
 *
 * You may want to implement this while implementing Task #3.
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetRootPageId() -> page_id_t { UNIMPLEMENTED("TODO(P2): Add implementation."); }

template class BPlusTree<GenericKey<4>, RID, GenericComparator<4>>;

template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 3>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 2>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 1>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, -1>;

template class BPlusTree<GenericKey<16>, RID, GenericComparator<16>>;

template class BPlusTree<GenericKey<32>, RID, GenericComparator<32>>;

template class BPlusTree<GenericKey<64>, RID, GenericComparator<64>>;

}  // namespace bustub
