#include "storage/page/b_plus_tree_leaf_page.h"
#include "common/config.h"
#include "gtest/gtest.h"
#include "storage/index/generic_key.h"
#include "test_util.h"  // NOLINT

namespace bustub {

using LeafPage = BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 3>;

auto key_schema = ParseCreateStatement("a bigint");
GenericComparator<8> comparator(key_schema.get());

auto PrepareLeafPage(char *page, int max_size, std::vector<int> &&keys) -> LeafPage * {
  auto *leaf = reinterpret_cast<LeafPage *>(page);

  leaf->Init(max_size);
  leaf->SetNextPageId(1);

  GenericKey<8> index_key;
  RID rid;

  for (auto key : keys) {
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }
  return leaf;
}

TEST(BPlusTreeLeafPage, RandomInsert) {
  char page[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page, 4, {});

  int64_t keys[] = {4, 1, 3, 2};

  for (auto i = 0; i < 4; i++) {
    auto key = keys[i];
    GenericKey<8> index_key;
    RID rid;
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);

    ASSERT_EQ(leaf->GetSize(), i + 1);
    if (i < 3) {
      ASSERT_EQ(leaf->IsFull(), false);
    } else {
      ASSERT_EQ(leaf->IsFull(), true);
    }
  }

  for (int64_t key = 1; key <= 4; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(leaf->Exist(index_key, comparator), true);
    ASSERT_EQ(comparator(leaf->KeyAt(key - 1), index_key), 0);
  }
}

TEST(BPlusTreeLeafPage, ReInsertWithTombstones) {
  char page[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page, 5, {1, 2});

  leaf->Remove(0);  // remove key 1

  GenericKey<8> index_key;
  RID rid;

  index_key.SetFromInteger(1);
  rid.Set(11, 11);
  leaf->Insert(index_key, rid, comparator);  // re-insert key 1

  ASSERT_EQ(leaf->GetSize(), 2);
  ASSERT_EQ(leaf->GetTombstones().size(), 0);

  auto ret = leaf->Lookup(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret->GetPageId(), 11);
}

TEST(BPlusTreeLeafPage, InsertWithTombstones) {
  char page[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page, 5, {1, 3});
  leaf->Init(4);

  GenericKey<8> index_key;
  RID rid;

  leaf->Remove(1);  // remove key 3

  // insert 2, should move tombstones
  index_key.SetFromInteger(2);
  rid.Set(2, 2);
  leaf->Insert(index_key, rid, comparator);

  index_key.SetFromInteger(3);
  auto ret = leaf->Lookup(index_key, comparator);
  ASSERT_FALSE(ret.has_value());

  // insert 4
  index_key.SetFromInteger(4);
  rid.Set(4, 4);
  leaf->Insert(index_key, rid, comparator);

  index_key.SetFromInteger(3);
  ret = leaf->Lookup(index_key, comparator);
  ASSERT_FALSE(ret.has_value());
}

TEST(BPlusTreeLeafPage, BasicSplit) {
  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page1, 5, {1, 2, 3, 4, 5});
  leaf->SetNextPageId(1);

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = PrepareLeafPage(page2, 5, {});

  leaf->Split(2, other);

  ASSERT_EQ(leaf->GetSize(), 2);
  ASSERT_EQ(leaf->GetNextPageId(), 2);
  for (int64_t key = 1; key <= 2; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(leaf->Exist(index_key, comparator), true);
    ASSERT_EQ(comparator(leaf->KeyAt(key - 1), index_key), 0);
  }

  ASSERT_EQ(other->GetSize(), 3);
  ASSERT_EQ(other->GetNextPageId(), 1);
  for (int64_t key = 3; key <= 5; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    ASSERT_EQ(other->Exist(index_key, comparator), true);
    ASSERT_EQ(comparator(other->KeyAt(key - 3), index_key), 0);
  }
}

TEST(BPlusTreeLeafPage, SplitWithTombstones) {
  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page1, 5, {0, 1, 2, 3, 4});
  leaf->SetNextPageId(1);

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = PrepareLeafPage(page2, 5, {});

  // remove 2 and 4
  leaf->Remove(1);
  leaf->Remove(3);

  leaf->Split(2, other);

  auto leaf_tombs = leaf->GetTombstones();
  ASSERT_EQ(leaf_tombs.size(), 1);
  ASSERT_EQ(leaf_tombs[0].GetAsInteger(), 1);

  auto other_tombs = other->GetTombstones();
  ASSERT_EQ(other_tombs.size(), 1);
  ASSERT_EQ(other_tombs[0].GetAsInteger(), 3);
}

TEST(BPlusTreeLeafPage, BasicLookup) {
  char page[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page, 5, {1, 2, 3, 4, 5});

  GenericKey<8> index_key;
  for (int64_t key = 1; key <= 5; ++key) {
    index_key.SetFromInteger(key);

    auto ret = leaf->Lookup(index_key, comparator);
    ASSERT_TRUE(ret.has_value());
    ASSERT_EQ(ret->GetPageId(), static_cast<int32_t>(key));
  }
}

TEST(BPlusTreeLeafPage, RandomRemove) {
  char page[BUSTUB_PAGE_SIZE];
  auto *leaf = PrepareLeafPage(page, 5, {1, 2, 3, 4, 5});

  // remove 1, in tombstone
  leaf->Remove(1);
  ASSERT_EQ(leaf->GetSize(), 5);
  ASSERT_EQ(leaf->GetTombstones().size(), 1);

  // remove 0, in tombstone
  leaf->Remove(0);
  ASSERT_EQ(leaf->GetSize(), 5);
  ASSERT_EQ(leaf->GetTombstones().size(), 2);

  // remove 2, in tombstone
  leaf->Remove(2);
  ASSERT_EQ(leaf->GetSize(), 5);
  ASSERT_EQ(leaf->GetTombstones().size(), 3);

  // remove 4, clean tombstone
  leaf->Remove(4);
  ASSERT_EQ(leaf->GetSize(), 1);
  ASSERT_EQ(leaf->GetTombstones().size(), 0);

  GenericKey<8> index_key;

  index_key.SetFromInteger(4);
  auto ret = leaf->Lookup(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret->GetPageId(), static_cast<int32_t>(4));
}

TEST(BPlusTreeLeafPage, LendToRight) {
  GenericKey<8> index_key;
  char page1[BUSTUB_PAGE_SIZE];
  auto left = PrepareLeafPage(page1, 5, {1, 2, 3, 4, 5});
  char page2[BUSTUB_PAGE_SIZE];
  auto right = PrepareLeafPage(page2, 5, {10});

  auto lend = left->LendToRight(right);
  index_key.SetFromInteger(5);
  ASSERT_EQ(comparator(lend, index_key), 0);

  ASSERT_EQ(left->GetSize(), 4);
  ASSERT_EQ(right->GetSize(), 2);

  ASSERT_FALSE(left->LookupIndex(lend, comparator).has_value());  // should remove from left
  ASSERT_TRUE(right->LookupIndex(lend, comparator).has_value());  // should add to right
  ASSERT_EQ(right->LookupIndex(lend, comparator).value(), 0);
}

TEST(BPlusTreeLeafPage, LendToLeft) {
  GenericKey<8> index_key;
  char page1[BUSTUB_PAGE_SIZE];
  auto left = PrepareLeafPage(page1, 5, {1});
  char page2[BUSTUB_PAGE_SIZE];
  auto right = PrepareLeafPage(page2, 5, {10, 11, 12, 13, 14});

  auto lend = right->LendToLeft(left);
  index_key.SetFromInteger(10);
  ASSERT_EQ(comparator(lend, index_key), 0);

  ASSERT_EQ(left->GetSize(), 2);
  ASSERT_EQ(right->GetSize(), 4);

  ASSERT_FALSE(right->LookupIndex(lend, comparator).has_value());  // should remove from left
  ASSERT_TRUE(left->LookupIndex(lend, comparator).has_value());    // should add to right
  ASSERT_EQ(left->LookupIndex(lend, comparator).value(), 1);
}

TEST(BPlusTreeLeafPage, BasicMerge) {
  char page1[BUSTUB_PAGE_SIZE];
  auto left = PrepareLeafPage(page1, 5, {1, 2, 3, 4});
  char page2[BUSTUB_PAGE_SIZE];
  auto right = PrepareLeafPage(page2, 5, {10});

  left->Merge(right);

  ASSERT_EQ(left->GetSize(), 5);
  ASSERT_EQ(right->GetSize(), 0);

  GenericKey<8> index_key;
  for (int64_t key = 1; key <= 4; ++key) {
    index_key.SetFromInteger(key);
    ASSERT_TRUE(left->LookupIndex(index_key, comparator).has_value());
  }

  index_key.SetFromInteger(10);
  ASSERT_TRUE(left->LookupIndex(index_key, comparator).has_value());
  ASSERT_EQ(left->LookupIndex(index_key, comparator).value(), 4);
}

TEST(BPlusTreeLeafPage, GetLowerBoundIndex) {
  char page1[BUSTUB_PAGE_SIZE];
  auto leaf = PrepareLeafPage(page1, 5, {1, 3, 5, 7});

  // 3 -> tombstone
  leaf->Remove(1);

  GenericKey<8> index_key;
  std::optional<size_t> ret;

  index_key.SetFromInteger(0);
  ret = leaf->GetLowerBoundIndex(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret.value(), 0);

  index_key.SetFromInteger(1);
  ret = leaf->GetLowerBoundIndex(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret.value(), 0);

  index_key.SetFromInteger(3);
  ret = leaf->GetLowerBoundIndex(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret.value(), 2);

  index_key.SetFromInteger(4);
  ret = leaf->GetLowerBoundIndex(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret.value(), 2);

  index_key.SetFromInteger(8);
  ret = leaf->GetLowerBoundIndex(index_key, comparator);
  ASSERT_FALSE(ret.has_value());
}

}  // namespace bustub
