#include "storage/page/b_plus_tree_leaf_page.h"
#include "common/config.h"
#include "gtest/gtest.h"
#include "storage/index/generic_key.h"
#include "test_util.h"  // NOLINT

namespace bustub {

using LeafPage = BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 3>;

TEST(BPlusTreeLeafPage, RandomInsert) {
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page[BUSTUB_PAGE_SIZE];

  auto *leaf = reinterpret_cast<LeafPage *>(page);
  leaf->Init(4);

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
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page[BUSTUB_PAGE_SIZE];

  auto *leaf = reinterpret_cast<LeafPage *>(page);
  leaf->Init(4);

  GenericKey<8> index_key;
  RID rid;

  index_key.SetFromInteger(1);
  rid.Set(1, 1);
  leaf->Insert(index_key, rid, comparator);

  index_key.SetFromInteger(2);
  rid.Set(2, 2);
  leaf->Insert(index_key, rid, comparator);

  leaf->Remove(0);  // remove key 1

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
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page[BUSTUB_PAGE_SIZE];

  auto *leaf = reinterpret_cast<LeafPage *>(page);
  leaf->Init(4);

  GenericKey<8> index_key;
  RID rid;

  index_key.SetFromInteger(1);
  rid.Set(1, 1);
  leaf->Insert(index_key, rid, comparator);

  index_key.SetFromInteger(3);
  rid.Set(3, 3);
  leaf->Insert(index_key, rid, comparator);

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
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = reinterpret_cast<LeafPage *>(page1);
  leaf->Init(5);
  leaf->SetNextPageId(1);

  for (auto key = 1; key <= 5; key++) {
    GenericKey<8> index_key;
    RID rid;
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = reinterpret_cast<LeafPage *>(page2);
  other->Init(5);

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
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = reinterpret_cast<LeafPage *>(page1);
  leaf->Init(5);
  leaf->SetNextPageId(1);

  GenericKey<8> index_key;
  RID rid;

  for (auto key = 0; key < 5; key++) {
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }

  // remove 2 and 4
  leaf->Remove(1);
  leaf->Remove(3);

  char page2[BUSTUB_PAGE_SIZE];
  auto *other = reinterpret_cast<LeafPage *>(page2);
  other->Init(5);

  leaf->Split(2, other);

  auto leaf_tombs = leaf->GetTombstones();
  ASSERT_EQ(leaf_tombs.size(), 1);
  ASSERT_EQ(leaf_tombs[0].GetAsInteger(), 1);

  auto other_tombs = other->GetTombstones();
  ASSERT_EQ(other_tombs.size(), 1);
  ASSERT_EQ(other_tombs[0].GetAsInteger(), 3);
}

TEST(BPlusTreeLeafPage, BasicLookup) {
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = reinterpret_cast<LeafPage *>(page1);
  leaf->Init(5);
  leaf->SetNextPageId(1);

  GenericKey<8> index_key;
  RID rid;

  for (auto key = 1; key <= 5; key++) {
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }

  for (int64_t key = 1; key <= 5; ++key) {
    GenericKey<8> index_key;
    index_key.SetFromInteger(key);

    auto ret = leaf->Lookup(index_key, comparator);
    ASSERT_TRUE(ret.has_value());
    ASSERT_EQ(ret->GetPageId(), static_cast<int32_t>(key));
  }
}

TEST(BPlusTreeLeafPage, RandomRemove) {
  auto key_schema = ParseCreateStatement("a bigint");
  GenericComparator<8> comparator(key_schema.get());

  char page1[BUSTUB_PAGE_SIZE];
  auto *leaf = reinterpret_cast<LeafPage *>(page1);
  leaf->Init(5);
  leaf->SetNextPageId(1);

  GenericKey<8> index_key;
  RID rid;

  for (auto key = 1; key <= 5; key++) {
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }

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

  index_key.SetFromInteger(4);
  auto ret = leaf->Lookup(index_key, comparator);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret->GetPageId(), static_cast<int32_t>(4));
}

}  // namespace bustub
