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
#include "common/logger.h"
#include "common/macros.h"
#include "storage/index/b_plus_tree_debug.h"
#include "storage/page/page_guard.h"

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
  auto header_page = guard.AsMut<BPlusTreeHeaderPage>();
  header_page->root_page_id_ = bpm_->NewPage();

  auto root_guard = bpm_->WritePage(header_page->root_page_id_, AccessType::Index);
  auto root_page = root_guard.AsMut<LeafPage>();
  root_page->Init(leaf_max_size_);
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
  auto page_guard = bpm_->ReadPage(header_page_id_);
  auto header_page = page_guard.As<BPlusTreeHeaderPage>();

  Context ctx;
  ctx.root_page_id_ = header_page->root_page_id_;
  ctx.guards_.push_back(std::move(page_guard));

  auto value = Lookup(ctx, key, header_page->root_page_id_);

  if (value.has_value()) {
    result->push_back(value.value());
    return true;
  }
  return false;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Lookup(Context &ctx, const KeyType &key, page_id_t page_id) -> std::optional<ValueType> {
  auto page_guard = bpm_->ReadPage(page_id);
  auto page = page_guard.As<BPlusTreePage>();

  ctx.guards_.clear();

  if (page->IsLeafPage()) {
    auto leaf_page = page_guard.As<LeafPage>();
    return leaf_page->Lookup(key, comparator_);
  }

  auto internal_page = page_guard.As<InternalPage>();
  ctx.guards_.push_back(std::move(page_guard));

  return Lookup(ctx, key, internal_page->Search(key, comparator_));
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
  Context ctx;
  InsertRet ret;

  // Optimistic Latch Crabbing Protocol
  Insert(ctx, key, value, ret);

  // Fallback to Pessimistic Latch Crabbing Protocol
  if (ret.success_ == BPlusTreeOpResult::OptimisticLockFailed) {
    ctx.Reset(false);
    ret.Clear();
    Insert(ctx, key, value, ret);
  }

  return ret.success_ == BPlusTreeOpResult::Success;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(Context &ctx, const KeyType &key, const ValueType &value, InsertRet &ret) -> void {
  PageGuard page_guard;
  if (ctx.IsOptimisticMode()) {
    page_guard = bpm_->ReadPage(header_page_id_);
  } else {
    page_guard = bpm_->WritePage(header_page_id_);
  }

  auto header_page = page_guard.AsMut<BPlusTreeHeaderPage>();
  ctx.SetRootPageId(header_page->root_page_id_);

  ctx.guards_.push_back(std::move(page_guard));
  Insert(ctx, key, value, header_page->root_page_id_, ret);

  if (ret.split_page_id_ == INVALID_PAGE_ID) {
    return;
  }
  SplitRootPage(ctx, ret, header_page);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::SplitRootPage(const Context &ctx, InsertRet &ret, BPlusTreeHeaderPage *header_page) -> void {
  BUSTUB_ENSURE(!ctx.IsOptimisticMode(), "Internal page split should only happen in pessimistic mode");
  BUSTUB_ENSURE(ret.success_ == BPlusTreeOpResult::Success,
                "Insert result should be success when internal page splits");

  auto [new_page_id, new_internal_page] = CreateNewPage<InternalPage>(internal_max_size_);
  new_internal_page->Init(ret.start_key_, header_page->root_page_id_, ret.split_page_id_);
  header_page->root_page_id_ = new_page_id;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(Context &ctx, const KeyType &key, const ValueType &value, page_id_t page_id, InsertRet &ret)
    -> void {
  PageGuard page_guard;
  if (ctx.IsOptimisticMode()) {
    page_guard = bpm_->ReadPage(page_id);
  } else {
    page_guard = bpm_->WritePage(page_id);
  }

  auto page = page_guard.AsMut<BPlusTreePage>();

  if (ctx.IsOptimisticMode() || page->CanReleaseAncestor(true)) {
    ctx.guards_.clear();
  }

  if (page->IsLeafPage()) {
    if (ctx.IsOptimisticMode()) {  // Upgrade to write lock
      page_guard.Drop();
      page_guard = bpm_->WritePage(page_id);
    }

    auto leaf_page = page_guard.AsMut<LeafPage>();
    InsertIntoLeafPage(ctx, key, value, leaf_page, ret);
    return;
  }

  auto internal_page = page_guard.AsMut<InternalPage>();
  ctx.guards_.push_back(std::move(page_guard));

  Insert(ctx, key, value, internal_page->Search(key, comparator_), ret);

  if (ret.split_page_id_ == INVALID_PAGE_ID) {
    return;
  }
  InsertIntoInternalPage(ctx, ret.start_key_, ret.split_page_id_, internal_page, ret);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::InsertIntoInternalPage(const Context &ctx, const KeyType &key, page_id_t page_id,
                                            InternalPage *page, InsertRet &ret) -> void {
  BUSTUB_ENSURE(!ctx.IsOptimisticMode(), "Internal page split should only happen in pessimistic mode");
  BUSTUB_ENSURE(ret.success_ == BPlusTreeOpResult::Success,
                "Insert result should be success when internal page splits");

  if (!page->IsFull()) {
    page->Insert(key, page_id, comparator_);
    ret.split_page_id_ = INVALID_PAGE_ID;
    return;
  }

  auto [new_page_id, new_internal_page] = CreateNewPage<InternalPage>(internal_max_size_);
  auto start_key = page->Split(new_page_id, new_internal_page);

  if (comparator_(key, start_key) < 0) {
    page->Insert(key, page_id, comparator_);
  } else {
    new_internal_page->Insert(key, page_id, comparator_);
  }

  ret.split_page_id_ = new_page_id;
  ret.start_key_ = start_key;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::InsertIntoLeafPage(Context &ctx, const KeyType &key, const ValueType &value, LeafPage *page,
                                        InsertRet &ret) -> void {
  if (page->Exist(key, comparator_)) {
    ret.success_ = BPlusTreeOpResult::Duplicate;
    return;
  }

  if (!page->IsFull()) {
    ret.success_ = BPlusTreeOpResult::Success;
    page->Insert(key, value, comparator_);
    return;
  }

  if (ctx.IsOptimisticMode()) {
    ret.success_ = BPlusTreeOpResult::OptimisticLockFailed;
    return;
  }

  auto [new_page_id, new_leaf_page] = CreateNewPage<LeafPage>(leaf_max_size_);
  page->Split(new_page_id, new_leaf_page);

  auto start_key = new_leaf_page->KeyAt(0);
  if (comparator_(key, start_key) < 0) {
    page->Insert(key, value, comparator_);
  } else {
    new_leaf_page->Insert(key, value, comparator_);
  }

  ret.start_key_ = start_key;
  ret.split_page_id_ = new_page_id;
  ret.success_ = BPlusTreeOpResult::Success;
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
  Context ctx;
  DeleteRet ret;

  // Optimistic Latch Crabbing Protocol
  Remove(ctx, key, ret);

  // Fallback to Pessimistic Latch Crabbing Protocol
  if (ret.success_ == BPlusTreeOpResult::OptimisticLockFailed) {
    ctx.Reset(false);
    ret.Clear();
    Remove(ctx, key, ret);
  }
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Remove(Context &ctx, const KeyType &key, DeleteRet &ret) -> void {
  PageGuard page_guard;
  if (ctx.IsOptimisticMode()) {
    page_guard = bpm_->ReadPage(header_page_id_);
  } else {
    page_guard = bpm_->WritePage(header_page_id_);
  }

  auto header_page = page_guard.AsMut<BPlusTreeHeaderPage>();
  ctx.SetRootPageId(header_page->root_page_id_);

  ctx.guards_.push_back(std::move(page_guard));
  Remove(ctx, key, header_page->root_page_id_, INVALID_PAGE_ID, false, ret);

  if (ret.split_page_id_ == INVALID_PAGE_ID) {
    return;
  }
  header_page->root_page_id_ = ret.split_page_id_;
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Remove(Context &ctx, const KeyType &key, page_id_t cur, page_id_t sibling, bool isLeftPage,
                            DeleteRet &ret) -> void {
  PageGuard guard;
  if (ctx.IsOptimisticMode()) {
    guard = bpm_->ReadPage(header_page_id_);
  } else {
    guard = bpm_->WritePage(header_page_id_);
  }
  auto page = guard.AsMut<BPlusTreePage>();

  if (ctx.IsOptimisticMode() || page->CanReleaseAncestor(false)) {
    ctx.guards_.clear();
  }

  if (page->IsLeafPage()) {
    guard.Drop();
    guard = bpm_->WritePage(header_page_id_);

    auto leaf_page = guard.AsMut<LeafPage>();
    return DeleteFromLeafPage(ctx, key, cur, leaf_page, sibling, isLeftPage, ret);
  }

  auto internal_page = guard.AsMut<InternalPage>();
  ctx.guards_.push_back(std::move(guard));

  CurAndSibling result;
  internal_page->SearchAndSibling(key, result);

  Remove(ctx, key, result.cur_, result.sibling_, result.is_left_, ret);

  if (ret.split_page_id_ == INVALID_PAGE_ID && ret.deleted_page_id_ == INVALID_PAGE_ID) {
    return;
  }

  if (ret.split_page_id_ != INVALID_PAGE_ID) {
    auto to_update = result.GetIndexByPageId(ret.split_page_id_);
    ret.split_page_id_ = INVALID_PAGE_ID;

    internal_page->Update(to_update, ret.start_key_, comparator_);
    return;
  }

  auto to_delete = result.GetIndexByPageId(ret.deleted_page_id_);
  ret.deleted_page_id_ = INVALID_PAGE_ID;

  DeleteFromInternalPage(ctx, to_delete, cur, internal_page, sibling, isLeftPage, ret);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::DeleteFromInternalPage(const Context &ctx, size_t to_delete, page_id_t cur_page_id,
                                            InternalPage *page, page_id_t sibling_page_id, bool isLeftPage,
                                            DeleteRet &ret) -> void {
  page->Remove(to_delete);

  if (!page->Underflow()) {
    return;
  }

  if (sibling_page_id == INVALID_PAGE_ID) {
    BUSTUB_ENSURE(ctx.IsRootPage(cur_page_id), "Only root page can have no sibling");

    LOG_DEBUG("[InternalPage] No sibling page for %d", cur);
    if (page->GetSize() == 1) {
      ret.split_page_id_ = cur_page_id;
    }
    return;
  }

  auto guard = bpm_->WritePage(sibling_page_id);
  auto sibling_page = guard.AsMut<InternalPage>();

  Balance<InternalPage>(page, sibling_page, cur_page_id, sibling_page_id, isLeftPage, ret);
}

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::DeleteFromLeafPage(Context &ctx, const KeyType &key, page_id_t cur, LeafPage *page,
                                        page_id_t sibling, bool isLeftPage, DeleteRet &ret) -> void {
  auto index = page->LookupIndex(key, comparator_);
  if (!index.has_value()) {
    ret.success_ = BPlusTreeOpResult::NotFound;
    return;
  }

  auto to_remove = index.value();
  if (ctx.IsOptimisticMode()) {
    if (page->CanSafeRemove(to_remove)) {
      page->SoftRemove(to_remove);

      ret.success_ = BPlusTreeOpResult::Success;
    } else {
      ret.success_ = BPlusTreeOpResult::OptimisticLockFailed;
    }
    return;
  }

  ret.success_ = BPlusTreeOpResult::Success;

  page->Remove(to_remove);
  if (!page->Underflow()) {
    return;
  }

  if (sibling == INVALID_PAGE_ID) {
    BUSTUB_ENSURE(ctx.IsRootPage(cur), "Only root page can have no sibling");
    LOG_DEBUG("[LeafPage] No sibling page for %d, skip deleting", cur);
    return;
  }

  auto guard = bpm_->WritePage(sibling);

  auto sibling_page = guard.AsMut<LeafPage>();
  sibling_page->CleanTombstones();

  Balance<LeafPage>(page, sibling_page, cur, sibling, isLeftPage, ret);
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
auto BPLUSTREE_TYPE::GetRootPageId() -> page_id_t {
  auto guard = bpm_->WritePage(header_page_id_);
  auto header_page = guard.AsMut<BPlusTreeHeaderPage>();
  return header_page->root_page_id_;
}

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
