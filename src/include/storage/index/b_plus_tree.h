//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// b_plus_tree.h
//
// Identification: src/include/storage/index/b_plus_tree.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

/**
 * b_plus_tree.h
 *
 * Implementation of simple b+ tree data structure where internal pages direct
 * the search and leaf pages contain actual data.
 * (1) We only support unique key
 * (2) support insert & remove
 * (3) The structure should shrink and grow dynamically
 * (4) Implement index iterator for range scan
 */
#pragma once

#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/config.h"
#include "storage/index/index_iterator.h"
#include "storage/page/b_plus_tree_header_page.h"
#include "storage/page/b_plus_tree_internal_page.h"
#include "storage/page/b_plus_tree_leaf_page.h"
#include "storage/page/page_guard.h"

namespace bustub {

struct PrintableBPlusTree;

enum BPlusTreeOpResult {
  Success = 1,
  Duplicate = 2,
  OptimisticLockFailed = 3,
  NotFound = 4,
};

template <typename KeyType>
struct BPlusTreeInsertRet {
  BPlusTreeOpResult success_;
  page_id_t split_page_id_{INVALID_PAGE_ID};
  KeyType start_key_;

  auto Clear() -> void {
    success_ = BPlusTreeOpResult::Duplicate;
    split_page_id_ = INVALID_PAGE_ID;
  }
};

template <typename KeyType>
struct BPlusTreeDeleteRet {
  BPlusTreeOpResult success_;
  KeyType start_key_;
  page_id_t split_page_id_{INVALID_PAGE_ID};
  page_id_t deleted_page_id_{INVALID_PAGE_ID};

  auto Clear() -> void {
    success_ = BPlusTreeOpResult::Duplicate;
    split_page_id_ = INVALID_PAGE_ID;
    deleted_page_id_ = INVALID_PAGE_ID;
  }
};

/**
 * @brief Definition of the Context class.
 *
 * Hint: This class is designed to help you keep track of the pages
 * that you're modifying or accessing.
 */
class Context {
 public:
  // Store the page guards of the pages that you're modifying here.
  std::deque<PageGuard> guards_;

  bool optimistic_mode_;

  page_id_t root_page_id_;

 public:
  Context() : optimistic_mode_(true), root_page_id_(INVALID_PAGE_ID) {}

  auto SetRootPageId(page_id_t page_id) -> void { root_page_id_ = page_id; }
  auto IsRootPage(page_id_t page_id) const -> bool { return page_id == root_page_id_; }
  auto IsOptimisticMode() const -> bool { return optimistic_mode_; }
  auto Reset(bool optimistic_mode) -> void {
    optimistic_mode_ = optimistic_mode;
    guards_.clear();
    root_page_id_ = INVALID_PAGE_ID;
  }
};

#define BPLUSTREE_TYPE BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>

// Main class providing the API for the Interactive B+ Tree.
FULL_INDEX_TEMPLATE_ARGUMENTS_DEFN
class BPlusTree {
  using InternalPage = BPlusTreeInternalPage<KeyType, page_id_t, KeyComparator>;
  using LeafPage = BPlusTreeLeafPage<KeyType, ValueType, KeyComparator>;
  using InsertRet = BPlusTreeInsertRet<KeyType>;
  using DeleteRet = BPlusTreeDeleteRet<KeyType>;

 public:
  explicit BPlusTree(std::string name, page_id_t header_page_id, BufferPoolManager *buffer_pool_manager,
                     const KeyComparator &comparator, int leaf_max_size = LEAF_PAGE_SLOT_CNT,
                     int internal_max_size = INTERNAL_PAGE_SLOT_CNT);

  // Returns true if this B+ tree has no keys and values.
  auto IsEmpty() const -> bool;

  // Insert a key-value pair into this B+ tree.
  auto Insert(const KeyType &key, const ValueType &value) -> bool;

  // Remove a key and its value from this B+ tree.
  void Remove(const KeyType &key);

  // Return the value associated with a given key
  auto GetValue(const KeyType &key, std::vector<ValueType> *result) -> bool;

  // Return the page id of the root node
  auto GetRootPageId() -> page_id_t;

  // Index iterator
  auto Begin() -> INDEXITERATOR_TYPE;

  auto End() -> INDEXITERATOR_TYPE;

  auto Begin(const KeyType &key) -> INDEXITERATOR_TYPE;

  void Print(BufferPoolManager *bpm);

  void Draw(BufferPoolManager *bpm, const std::filesystem::path &outf);

  auto DrawBPlusTree() -> std::string;

  // read data from file and insert one by one
  void InsertFromFile(const std::filesystem::path &file_name);

  // read data from file and remove one by one
  void RemoveFromFile(const std::filesystem::path &file_name);

  void BatchOpsFromFile(const std::filesystem::path &file_name);

  // Do not change this type to a BufferPoolManager!
  std::shared_ptr<TracedBufferPoolManager> bpm_;

 private:
  void ToGraph(page_id_t page_id, const BPlusTreePage *page, std::ofstream &out);

  void PrintTree(page_id_t page_id, const BPlusTreePage *page);

  auto ToPrintableBPlusTree(page_id_t root_id) -> PrintableBPlusTree;

  // member variable
  std::string index_name_;
  KeyComparator comparator_;
  std::vector<std::string> log;  // NOLINT
  int leaf_max_size_;
  int internal_max_size_;
  page_id_t header_page_id_;

 private:
  auto Insert(Context &ctx, const KeyType &key, const ValueType &value, InsertRet &ret) -> void;
  auto Insert(Context &ctx, const KeyType &key, const ValueType &value, page_id_t page_id, InsertRet &ret) -> void;
  auto InsertIntoLeafPage(Context &ctx, const KeyType &key, const ValueType &value, LeafPage *page, InsertRet &ret)
      -> void;
  auto InsertIntoInternalPage(const Context &ctx, const KeyType &key, page_id_t page_id, InternalPage *page,
                              InsertRet &ret) -> void;
  auto SplitRootPage(const Context &ctx, InsertRet &ret, BPlusTreeHeaderPage *header_page) -> void;
  auto Lookup(Context &ctx, const KeyType &key, page_id_t page_id) -> std::optional<ValueType>;
  auto DeleteFromLeafPage(Context &ctx, const KeyType &key, page_id_t cur, LeafPage *page, page_id_t sibling,
                          bool isLeftPage, DeleteRet &ret) -> void;
  auto Remove(Context &ctx, const KeyType &key, page_id_t cur, page_id_t sibling, bool isLeftPage, DeleteRet &ret)
      -> void;
  auto DeleteFromInternalPage(const Context &ctx, size_t to_delete, page_id_t cur_page_id, InternalPage *page,
                              page_id_t sibling_page_id, bool isLeftPage, DeleteRet &ret) -> void;
  auto Remove(Context &ctx, const KeyType &key, DeleteRet &ret) -> void;

  template <typename T>
  auto CreateNewPage(int max_size) -> std::pair<page_id_t, T *> {
    auto new_page_id = bpm_->NewPage();
    auto guard = bpm_->WritePage(new_page_id, AccessType::Index);

    auto new_page = guard.AsMut<T>();
    new_page->Init(max_size);

    return {new_page_id, new_page};
  }

  template <typename T>
  auto Redistribute(T *page, T *sibling_page, page_id_t cur_page_id, page_id_t sibling_page_id, bool isLeftPage,
                    DeleteRet &ret) -> void {
    if (!isLeftPage) {
      return Redistribute(sibling_page, page, sibling_page_id, cur_page_id, true, ret);
    }

    ret.start_key_ = sibling_page->Lend(page);
    ret.split_page_id_ = cur_page_id;
    ret.deleted_page_id_ = INVALID_PAGE_ID;
  }

  template <typename T>
  auto Merge(T *page, T *sibling_page, page_id_t cur_page_id, page_id_t sibling_page_id, bool isLeftPage,
             DeleteRet &ret) -> void {
    if (!isLeftPage) {
      return Merge(sibling_page, page, sibling_page_id, cur_page_id, true, ret);
    }

    sibling_page->Merge(page);
    ret.deleted_page_id_ = cur_page_id;
    ret.split_page_id_ = INVALID_PAGE_ID;
  }

  template <typename T>
  auto Balance(T *page, T *sibling_page, page_id_t cur_page_id, page_id_t sibling_page_id, bool isLeftPage,
               DeleteRet &ret) -> void {
    if (sibling_page->CanLendAKey()) {
      return Redistribute<T>(page, sibling_page, cur_page_id, sibling_page_id, isLeftPage, ret);
    }

    Merge<T>(page, sibling_page, cur_page_id, sibling_page_id, isLeftPage, ret);
  }
};

/**
 * @brief for test only. PrintableBPlusTree is a printable B+ tree.
 * We first convert B+ tree into a printable B+ tree and the print it.
 */
struct PrintableBPlusTree {
  int size_;
  std::string keys_;
  std::vector<PrintableBPlusTree> children_;

  /**
   * @brief BFS traverse a printable B+ tree and print it into
   * into out_buf
   *
   * @param out_buf
   */
  void Print(std::ostream &out_buf) {
    std::vector<PrintableBPlusTree *> que = {this};
    while (!que.empty()) {
      std::vector<PrintableBPlusTree *> new_que;

      for (auto &t : que) {
        int padding = (t->size_ - t->keys_.size()) / 2;
        out_buf << std::string(padding, ' ');
        out_buf << t->keys_;
        out_buf << std::string(padding, ' ');

        for (auto &c : t->children_) {
          new_que.push_back(&c);
        }
      }
      out_buf << "\n";
      que = new_que;
    }
  }
};

}  // namespace bustub
