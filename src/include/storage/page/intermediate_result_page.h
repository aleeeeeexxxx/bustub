#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "storage/table/tuple.h"

namespace bustub {

/**
 * Page to hold the intermediate data for external merge sort and hash join.
 * Supports variable-length tuples.
 */
class IntermediateResultPage {
 public:
  auto CanInsert(const Tuple &tuple) -> bool {
    return offset_ + tuple.GetSerializedSize() <= BUSTUB_PAGE_SIZE - sizeof(size_t);
  }

  auto InsertTuple(const Tuple &tuple) -> void {
    tuple.SerializeTo(data_ + offset_);
    offset_ += tuple.GetSerializedSize();
  }

  auto ToTuples(std::vector<Tuple> &tuples) const -> void {
    size_t cur = 0;
    Tuple temp;
    while (cur < offset_) {
      temp.DeserializeFrom(data_ + cur);

      cur += temp.GetSerializedSize();
      tuples.push_back(temp);
    }

    BUSTUB_ASSERT(cur == offset_, "Deserialized size does not match offset");
  }

  auto Reset() -> void { offset_ = 0; }

 private:
  size_t offset_{0};
  char data_[0];
};

}  // namespace bustub
