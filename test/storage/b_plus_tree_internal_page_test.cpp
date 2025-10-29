#include "storage/page/b_plus_tree_internal_page.h"
#include "gtest/gtest.h"
#include "storage/index/generic_key.h"
#include "test_util.h"  // NOLINT

namespace bustub {

using InternalPage = BPlusTreeInternalPage<GenericKey<8>, page_id_t, GenericComparator<8>>;

TEST(BPlusTreeInternalPage, RandomInsert) {
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page[BUSTUB_PAGE_SIZE];

  auto *internal = reinterpret_cast<InternalPage *>(page);
  internal->Init(6);

  GenericKey<8> first_key;
  first_key.SetFromInteger(1);
  internal->Init(first_key, 0, 1);

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
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *internal = reinterpret_cast<InternalPage *>(page1);
  internal->Init(6);

  GenericKey<8> first_key;
  first_key.SetFromInteger(1);
  internal->Init(first_key, 0, 1);

  for (auto key = 2; key <= 5; key++) {
    GenericKey<8> index_key;
    int64_t value = key & 0xFFFFFFFF;

    index_key.SetFromInteger(key);

    internal->Insert(index_key, value, comparator);
  }

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
  for (int64_t key = 4; key <= 5; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(comparator(other->KeyAt(key - 3), index_key), 0);
    ASSERT_EQ(other->ValueAt(key - 3), key);
  }
}

TEST(BPlusTreeInternalPage, BasicSearch) {
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *internal = reinterpret_cast<InternalPage *>(page1);
  internal->Init(3);

  GenericKey<8> key;
  key.SetFromInteger(1);
  internal->Init(key, 0, 1);

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

}  // namespace bustub
