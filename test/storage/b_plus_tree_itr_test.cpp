#include <memory>
#include "common/config.h"
#include "gtest/gtest.h"
#include "storage/index/generic_key.h"
#include "storage/index/index_iterator.h"
#include "storage/page/b_plus_tree_leaf_page.h"
#include "test_util.h"

namespace bustub {

using LeafPage = BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>, 3>;
using Itr = IndexIterator<GenericKey<8>, RID, GenericComparator<8>, 3>;

static std::filesystem::path db_fname("test_itr.bustub");

// The number of frames we give to the buffer pool.
const size_t FRAMES = 10;

auto key_schema = ParseCreateStatement("a bigint");
GenericComparator<8> comparator(key_schema.get());

GenericKey<8> index_key;

auto PrepareLeafPage(const std::shared_ptr<TracedBufferPoolManager> &bpm, int max_size, std::vector<int> &&keys,
                     page_id_t next_page_id) -> page_id_t {
  auto new_page_id = bpm->NewPage();
  auto page_guard = bpm->WritePage(new_page_id);
  auto leaf = page_guard.AsMut<LeafPage>();

  leaf->Init(max_size);
  leaf->SetNextPageId(next_page_id);

  RID rid;

  for (auto key : keys) {
    int64_t value = key & 0xFFFFFFFF;

    rid.Set(static_cast<int32_t>(key), value);
    index_key.SetFromInteger(key);

    leaf->Insert(index_key, rid, comparator);
  }
  return new_page_id;
}

TEST(BPlusTreeItr, Iterator) {
  auto disk_manager = std::make_unique<DiskManager>(db_fname);
  auto bpm = std::make_unique<BufferPoolManager>(FRAMES, disk_manager.get());
  auto traced_bpm = std::make_shared<TracedBufferPoolManager>(bpm.get());

  auto page2 = PrepareLeafPage(traced_bpm, 5, {21, 23}, INVALID_PAGE_ID);
  auto page1 = PrepareLeafPage(traced_bpm, 5, {11, 13, 15, 17, 19}, page2);

  auto guard = bpm->ReadPage(page1);

  Itr itr(page1, 2, std::move(guard), traced_bpm, comparator);

  std::vector<int> expected_keys = {15, 17, 19, 21, 23};
  for (auto key : expected_keys) {
    ASSERT_FALSE(itr.IsEnd());

    auto [k, v] = *itr;
    ASSERT_EQ(key, v.GetPageId());

    ++itr;
  }

  ASSERT_TRUE(itr.IsEnd());
}

TEST(BPlusTreeItr, IteratorWithTombstones) {
  auto disk_manager = std::make_unique<DiskManager>(db_fname);
  auto bpm = std::make_unique<BufferPoolManager>(FRAMES, disk_manager.get());
  auto traced_bpm = std::make_shared<TracedBufferPoolManager>(bpm.get());

  auto page2 = PrepareLeafPage(traced_bpm, 5, {21, 23}, INVALID_PAGE_ID);
  auto page1 = PrepareLeafPage(traced_bpm, 5, {11, 13, 15, 17, 19}, page2);

  auto guard1 = bpm->ReadPage(page1);
  auto leaf1 = guard1.AsMut<LeafPage>();
  leaf1->Remove(3);

  auto guard2 = bpm->ReadPage(page2);
  auto leaf2 = guard2.AsMut<LeafPage>();
  leaf2->Remove(1);

  Itr itr(page1, 2, std::move(guard1), traced_bpm, comparator);

  std::vector<int> expected_keys = {15, 19, 21};
  for (auto key : expected_keys) {
    ASSERT_FALSE(itr.IsEnd());

    auto [k, v] = *itr;
    ASSERT_EQ(key, v.GetPageId());

    ++itr;
  }

  ASSERT_TRUE(itr.IsEnd());
}

}  // namespace bustub