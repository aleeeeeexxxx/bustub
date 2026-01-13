#include "storage/page/intermediate_result_page.h"
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "gtest/gtest.h"
#include "storage/disk/disk_manager_memory.h"

namespace bustub {

auto Equal(const Tuple &a, const Tuple &b) -> bool {
  if (a.GetLength() != b.GetLength()) {
    return false;
  }

  auto a_ptr = a.GetData();
  auto b_ptr = b.GetData();
  for (uint32_t i = 0; i < a.GetLength(); i++) {
    if (a_ptr[i] != b_ptr[i]) {
      return false;
    }
  }
  return true;
}

TEST(IntermediateResultPageTest, BasicInsertAndRetrieve) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  char buf1[100] = {1};
  Tuple t1{RID{}, buf1, 20};
  char buf2[100] = {2};
  Tuple t2{RID{}, buf2, 20};
  char buf3[100] = {3};
  Tuple t3{RID{}, buf3, 20};

  std::vector<Tuple> expected = {t1, t2, t3};

  auto page_id = bpm->NewPage();

  // insert
  auto guard = bpm->WritePage(page_id);
  auto page = guard.AsMut<IntermediateResultPage>();
  for (auto &t : expected) {
    EXPECT_EQ(page->CanInsert(t), true);
    page->InsertTuple(t);
  }

  // release the page
  guard.Drop();

  // flush the old one into dish
  auto page_id_2 = bpm->NewPage();
  guard = bpm->WritePage(page_id_2);
  guard.Drop();

  // read back
  guard = bpm->WritePage(page_id);
  page = guard.AsMut<IntermediateResultPage>();

  std::vector<Tuple> tuples;
  page->ToTuples(tuples);

  EXPECT_EQ(tuples.size(), 3);

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_TRUE(Equal(tuples[i], expected[i]));
  }

  delete bpm;
}

TEST(IntermediateResultPageTest, Overflow) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto page_id = bpm->NewPage();

  // insert
  auto guard = bpm->WritePage(page_id);
  auto page = guard.AsMut<IntermediateResultPage>();

  char buf[1000] = {1};
  Tuple t{RID{}, buf, 1000};

  // ( 8192 - 8 ) / 1000 = 8.184
  for (size_t i = 1; i <= 8; i++) {
    EXPECT_TRUE(page->CanInsert(t));
    page->InsertTuple(t);
  }

  EXPECT_FALSE(page->CanInsert(t));

  delete bpm;
}

TEST(IntermediateResultPageTest, Reset) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto page_id = bpm->NewPage();

  // insert
  auto guard = bpm->WritePage(page_id);
  auto page = guard.AsMut<IntermediateResultPage>();

  size_t cnt = 0;
  char buf[1000] = {1};
  Tuple t{RID{}, buf, 1000};

  while (page->CanInsert(t)) {
    page->InsertTuple(t);
    cnt++;
  }

  page->Reset();

  std::vector<Tuple> tuples;
  page->ToTuples(tuples);
  EXPECT_EQ(tuples.size(), 0);

  for (size_t i = 1; i <= cnt; i++) {
    EXPECT_TRUE(page->CanInsert(t));
    page->InsertTuple(t);
  }

  delete bpm;
}
}  // namespace bustub