#include "storage/page/b_plus_tree_internal_page.h"
#include <cassert>
#include "gtest/gtest.h"
#include "storage/index/generic_key.h"
#include "test_util.h"  // NOLINT

namespace bustub {

using InternalPage = BPlusTreeInternalPage<GenericKey<8>, page_id_t, GenericComparator<8>>;

auto key_schema = ParseCreateStatement("a bigint");
GenericComparator<8> comparator(key_schema.get());

auto PrepareInternalPage(char *page, int max_size, std::vector<int> &&keys) -> InternalPage * {
  auto *internal = reinterpret_cast<InternalPage *>(page);
  internal->Init(max_size);

  GenericKey<8> index_key;

  index_key.SetFromInteger(keys[0]);
  internal->SetKeyAt(0, index_key);
  index_key.SetFromInteger(keys[1]);
  internal->Init(index_key, keys[0], keys[1]);

  for (size_t i = 2; i < keys.size(); ++i) {
    int64_t value = keys[i] & 0xFFFFFFFF;
    index_key.SetFromInteger(keys[i]);

    internal->Insert(index_key, value, comparator);
  }
  return internal;
}

TEST(BPlusTreeInternalPage, RandomInsert) {
  char page[BUSTUB_PAGE_SIZE];
  auto internal = PrepareInternalPage(page, 6, {0, 1});

  int64_t keys[] = {5, 2, 4, 3};

  for (auto i = 0; i < 4; i++) {
    auto key = keys[i];
    GenericKey<8> index_key;
    int64_t value = key & 0xFFFFFFFF;

    index_key.SetFromInteger(key);

    internal->Insert(index_key, value, comparator);

    ASSERT_EQ(internal->GetSize(), i + 3);
    if (i < 3) {
      ASSERT_EQ(internal->IsFull(), false);
    } else {
      ASSERT_EQ(internal->IsFull(), true);
    }
  }

  for (int64_t key = 1; key <= 5; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(comparator(internal->KeyAt(key), index_key), 0);
    ASSERT_EQ(internal->ValueAt(key), key);
  }
}

TEST(BPlusTreeInternalPage, BasicSplit) {
  char page1[BUSTUB_PAGE_SIZE];
  auto internal = PrepareInternalPage(page1, 6, {0, 1, 2, 3, 4, 5});

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = reinterpret_cast<InternalPage *>(page2);
  other->Init(5);

  auto key = internal->Split(2, other);
  GenericKey<8> split_key;
  split_key.SetFromInteger(3);
  ASSERT_EQ(comparator(key, split_key), 0);

  ASSERT_EQ(internal->GetSize(), 3);
  ASSERT_EQ(internal->ValueAt(0), 0);
  for (int64_t key = 1; key <= 2; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(comparator(internal->KeyAt(key), index_key), 0);
    ASSERT_EQ(internal->ValueAt(key), key);
  }

  ASSERT_EQ(other->GetSize(), 3);
  ASSERT_EQ(other->ValueAt(0), 3);
  for (int64_t key = 3; key <= 5; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(comparator(other->KeyAt(key - 3), index_key), 0);
    ASSERT_EQ(other->ValueAt(key - 3), key);
  }
}

TEST(BPlusTreeInternalPage, BasicSearch) {
  char page[BUSTUB_PAGE_SIZE];
  auto *internal = PrepareInternalPage(page, 6,
                                       {
                                           0,
                                           1,
                                       });

  GenericKey<8> key;
  key.SetFromInteger(4);
  internal->Insert(key, 2, comparator);

  /*
   * Search
   */

  key.SetFromInteger(0);
  ASSERT_EQ(internal->Search(key, comparator), 0);

  key.SetFromInteger(1);
  ASSERT_EQ(internal->Search(key, comparator), 1);

  key.SetFromInteger(2);
  ASSERT_EQ(internal->Search(key, comparator), 1);

  key.SetFromInteger(4);
  ASSERT_EQ(internal->Search(key, comparator), 2);

  key.SetFromInteger(5);
  ASSERT_EQ(internal->Search(key, comparator), 2);
}

TEST(BPlusTreeInternalPage, SearchSibling) {
  char page[BUSTUB_PAGE_SIZE];
  auto *internal = PrepareInternalPage(page, 6, {0, 1, 2});

  CurAndSibling result;
  GenericKey<8> key;

  key.SetFromInteger(2);
  internal->SearchCurrentAndSibling(key, result, comparator);
  ASSERT_EQ(result.cur_, 2);
  ASSERT_EQ(result.sibling_, 1);
  ASSERT_EQ(result.is_left_, true);

  key.SetFromInteger(0);
  internal->SearchCurrentAndSibling(key, result, comparator);
  ASSERT_EQ(result.cur_, 0);
  ASSERT_EQ(result.sibling_, 1);
  ASSERT_EQ(result.is_left_, false);
}

TEST(BPlusTreeInternalPage, Lend) {
  char page1[BUSTUB_PAGE_SIZE];
  auto *internal = PrepareInternalPage(page1, 6, {0, 1, 2, 3, 4});

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = PrepareInternalPage(page2, 6, {5, 6});

  GenericKey<8> key;

  auto lend = internal->Lend(other);
  key.SetFromInteger(4);
  ASSERT_EQ(comparator(lend, key), 0);

  ASSERT_EQ(internal->Search(key, comparator), 3);
  ASSERT_EQ(other->Search(key, comparator), 4);
}

TEST(BPlusTreeInternalPage, Merge) {
  char page[BUSTUB_PAGE_SIZE];
  auto *internal = PrepareInternalPage(page, 3,
                                       {
                                           0,
                                           1,
                                           2,
                                       });

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = PrepareInternalPage(page2, 6, {5, 6});

  internal->Merge(other);
  ASSERT_EQ(internal->GetSize(), 5);
  ASSERT_EQ(other->GetSize(), 0);

  GenericKey<8> key;
  key.SetFromInteger(5);
  ASSERT_EQ(internal->Search(key, comparator), 5);
  key.SetFromInteger(6);
  ASSERT_EQ(internal->Search(key, comparator), 6);
}

TEST(BPlusTreeInternalPage, Remove) {
  char page[BUSTUB_PAGE_SIZE];
  auto *internal = PrepareInternalPage(page, 6, {0, 1, 2, 3, 4});

  GenericKey<8> key;

  key.SetFromInteger(2);
  internal->Remove(2);
  ASSERT_EQ(internal->GetSize(), 4);
}

}  // namespace bustub
