#include "execution/executors/external_merge_sort_executor.h"
#include <memory>
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "catalog/column.h"
#include "execution/execution_common.h"
#include "execution/expressions/column_value_expression.h"
#include "gtest/gtest.h"
#include "storage/disk/disk_manager_memory.h"
#include "storage/page/intermediate_result_page.h"
#include "type/value_factory.h"

namespace bustub {

auto RunTest(BufferPoolManager *bpm, Schema &schema, std::vector<std::vector<Tuple>> data,
             const std::vector<OrderBy> &order_bys, std::function<void(const std::vector<Tuple> &)> validate) -> void {
  size_t total_cnt = 0;

  // insert into page
  std::vector<page_id_t> page_ids;

  for (auto &run : data) {
    auto page_id = bpm->NewPage();
    auto guard = bpm->WritePage(page_id);
    auto page = guard.AsMut<IntermediateResultPage<Tuple>>();

    for (auto &tuple : run) {
      EXPECT_TRUE(page->CanInsert(tuple));
      page->Insert(tuple);
      total_cnt++;
    }

    guard.Drop();
    page_ids.push_back(page_id);
  }

  // sort
  TupleComparator cmp{order_bys, schema};
  MergeSortRun merge_sort_run(bpm, cmp);

  auto expected = merge_sort_run.Sort(page_ids);

  // validation
  std::vector<Tuple> sorted_tuples;

  for (auto page_id : expected) {
    auto guard = bpm->WritePage(page_id);
    auto page = guard.AsMut<IntermediateResultPage<Tuple>>();

    page->ReadAll(sorted_tuples);
  }

  EXPECT_EQ(sorted_tuples.size(), total_cnt);
  validate(sorted_tuples);
}

TEST(MergeSortRunTest, SortSinglePage) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto col1 = Column{"col1", TypeId::INTEGER};
  auto col2 = Column{"col2", TypeId::VARCHAR, 256};
  auto schema = Schema({col1, col2});

  // prepare test data
  std::vector<Tuple> tuples{
      Tuple{{ValueFactory::GetIntegerValue(9), ValueFactory::GetVarcharValue("9")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(2), ValueFactory::GetVarcharValue("2")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(8), ValueFactory::GetVarcharValue("8")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(3), ValueFactory::GetVarcharValue("3")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(5), ValueFactory::GetVarcharValue("5")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(6), ValueFactory::GetVarcharValue("6")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetVarcharValue("1")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("0")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(7), ValueFactory::GetVarcharValue("7")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(4), ValueFactory::GetVarcharValue("4")}, &schema},
  };

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i);
        }
      });

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 1, col2)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          auto expected = 9 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
      });

  delete bpm;
}

TEST(MergeSortRunTest, SortSinglePageWithNulls) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto col1 = Column{"col1", TypeId::INTEGER};
  auto col2 = Column{"col2", TypeId::VARCHAR, 256};
  auto schema = Schema({col1, col2});

  // prepare test data
  std::vector<Tuple> tuples{
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("17")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("18")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(9), ValueFactory::GetVarcharValue("9")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(2), ValueFactory::GetVarcharValue("2")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("19")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(8), ValueFactory::GetVarcharValue("8")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(3), ValueFactory::GetVarcharValue("3")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("10")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(5), ValueFactory::GetVarcharValue("5")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(6), ValueFactory::GetVarcharValue("6")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("11")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetVarcharValue("1")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("16")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("0")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("13")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(7), ValueFactory::GetVarcharValue("7")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("14")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("15")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(4), ValueFactory::GetVarcharValue("4")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("12")}, &schema},
  };

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i - 10);
        }
      });

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_LAST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i);
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
      });

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
        for (size_t i = 10; i < 20; i++) {
          auto expected = 19 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
      });

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_LAST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          auto expected = 9 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
      });

  delete bpm;
}

TEST(MergeSortRunTest, SortMultiplePages) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto col1 = Column{"col1", TypeId::INTEGER};
  auto col2 = Column{"col2", TypeId::VARCHAR, 256};
  auto schema = Schema({col1, col2});

  // prepare test data
  std::vector<Tuple> tuples1{
      Tuple{{ValueFactory::GetIntegerValue(9), ValueFactory::GetVarcharValue("9")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(2), ValueFactory::GetVarcharValue("2")}, &schema},
  };

  std::vector<Tuple> tuples2{
      Tuple{{ValueFactory::GetIntegerValue(8), ValueFactory::GetVarcharValue("8")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(3), ValueFactory::GetVarcharValue("3")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(5), ValueFactory::GetVarcharValue("5")}, &schema},
  };

  std::vector<Tuple> tuples3{
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("0")}, &schema},
  };

  std::vector<Tuple> tuples4{
      Tuple{{ValueFactory::GetIntegerValue(7), ValueFactory::GetVarcharValue("7")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(4), ValueFactory::GetVarcharValue("4")}, &schema},
  };

  std::vector<Tuple> tuples5{
      Tuple{{ValueFactory::GetIntegerValue(6), ValueFactory::GetVarcharValue("6")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetVarcharValue("1")}, &schema},
  };

  std::vector<std::vector<Tuple>> tuples{tuples1, tuples2, tuples3, tuples4, tuples5};

  RunTest(
      bpm, schema, tuples,
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i);
        }
      });

  RunTest(
      bpm, schema, tuples,
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 1, col2)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          auto expected = 9 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
      });

  delete bpm;
}

TEST(MergeSortRunTest, SortMultiplePageWithNulls) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto col1 = Column{"col1", TypeId::INTEGER};
  auto col2 = Column{"col2", TypeId::VARCHAR, 256};
  auto schema = Schema({col1, col2});

  // prepare test data
  std::vector<Tuple> tuples1{
      Tuple{{ValueFactory::GetIntegerValue(6), ValueFactory::GetVarcharValue("6")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("11")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetVarcharValue("1")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("0")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("13")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(7), ValueFactory::GetVarcharValue("7")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("14")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("15")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(4), ValueFactory::GetVarcharValue("4")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("16")}, &schema},
  };

  std::vector<Tuple> tuples2{
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("17")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("18")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(9), ValueFactory::GetVarcharValue("9")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(2), ValueFactory::GetVarcharValue("2")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("19")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(8), ValueFactory::GetVarcharValue("8")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(3), ValueFactory::GetVarcharValue("3")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("10")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(5), ValueFactory::GetVarcharValue("5")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("12")}, &schema},
  };

  RunTest(
      bpm, schema, {tuples1, tuples2},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i - 10);
        }
      });

  RunTest(
      bpm, schema, {tuples1, tuples2},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_LAST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), i);
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
      });

  RunTest(
      bpm, schema, {tuples1, tuples2},
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
        for (size_t i = 10; i < 20; i++) {
          auto expected = 19 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
      });

  RunTest(
      bpm, schema, {tuples1, tuples2},
      {
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_LAST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        for (size_t i = 0; i < 10; i++) {
          auto expected = 9 - i;
          EXPECT_EQ(sorted_tuples[i].GetValue(&schema, 0).GetAs<int>(), expected);
        }
        for (size_t i = 10; i < 20; i++) {
          EXPECT_TRUE(sorted_tuples[i].GetValue(&schema, 0).IsNull());
        }
      });

  delete bpm;
}

TEST(MergeSortRunTest, SortMutipleColumns) {
  auto disk_manager = std::make_unique<DiskManagerUnlimitedMemory>();
  auto *bpm = new BufferPoolManager(1, disk_manager.get());

  auto col1 = Column{"col1", TypeId::INTEGER};
  auto col2 = Column{"col2", TypeId::VARCHAR, 256};
  auto schema = Schema({col1, col2});

  // prepare test data
  std::vector<Tuple> tuples{
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetNullValueByType(TypeId::VARCHAR)}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("11")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(1), ValueFactory::GetVarcharValue("1")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("01")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("13")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetVarcharValue("02")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("14")}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("15")}, &schema},
      Tuple{{ValueFactory::GetIntegerValue(0), ValueFactory::GetNullValueByType(TypeId::VARCHAR)}, &schema},
      Tuple{{ValueFactory::GetNullValueByType(TypeId::INTEGER), ValueFactory::GetVarcharValue("16")}, &schema},
  };

  std::vector<std::string> expected{"(<NULL>, 16)", "(<NULL>, 15)", "(<NULL>, 14)", "(<NULL>, 13)", "(<NULL>, 11)",
                                    "(0, 02)",      "(0, 01)",      "(0, <NULL>)",  "(1, 1)",       "(1, <NULL>)"};

  RunTest(
      bpm, schema, {tuples},
      {
          OrderBy{OrderByType::ASC, OrderByNullType::NULLS_FIRST, std::make_shared<ColumnValueExpression>(0, 0, col1)},
          OrderBy{OrderByType::DESC, OrderByNullType::NULLS_LAST, std::make_shared<ColumnValueExpression>(0, 1, col2)},
      },
      [&](const std::vector<Tuple> &sorted_tuples) {
        std::vector<std::string> actual;
        for (auto &tuple : sorted_tuples) {
          actual.push_back(tuple.ToString(&schema));
        }
        EXPECT_EQ(actual, expected);
      });

  delete bpm;
}

}  // namespace bustub
