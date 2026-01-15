#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

#include "buffer/buffer_pool_manager.h"
#include "common/macros.h"
#include "storage/page/page.h"
#include "storage/table/tuple.h"

namespace bustub {

template <typename, typename = void>
struct serializable : std::false_type {};

template <typename T>
struct serializable<T, std::void_t<decltype(std::declval<T>().SerializeTo(static_cast<char *>(nullptr)))>>
    : std::true_type {};

template <typename, typename = void>
struct deserializable : std::false_type {};

template <typename T>
struct deserializable<T, std::void_t<decltype(std::declval<T>().DeserializeFrom(static_cast<char *>(nullptr)))>>
    : std::true_type {};

template <typename, typename = void>
struct countable : std::false_type {};

template <typename T>
struct countable<T, std::void_t<decltype(std::declval<T>().GetSerializedSize())>>
    : std::is_same<decltype(std::declval<T>().GetSerializedSize()), uint32_t> {};

template <typename T>
struct is_intermedate_result
    : std::bool_constant<serializable<T>::value && deserializable<T>::value && countable<T>::value> {};

template <typename T>
class IntermediateResult {
  static_assert(is_intermedate_result<T>::value, "T must be IntermediateResult");
};
/**
 * Page to hold the intermediate data for external merge sort and hash join.
 * Supports variable-length tuples.
 */
template <typename T>
class IntermediateResultPage : IntermediateResult<T> {
 public:
  auto CanInsert(const T &item) -> bool {
    return offset_ + item.GetSerializedSize() <= BUSTUB_PAGE_SIZE - sizeof(size_t);
  };

  auto Insert(const T &item) -> void {
    item.SerializeTo(data_ + offset_);
    offset_ += item.GetSerializedSize();
  };

  auto ReadAll(std::vector<T> &results) const -> void {
    size_t cur = 0;
    T temp;
    while (cur < offset_) {
      temp.DeserializeFrom(data_ + cur);

      cur += temp.GetSerializedSize();
      results.push_back(temp);
    }

    BUSTUB_ASSERT(cur == offset_, "Deserialized size does not match offset");
  };

  auto Reset() -> void { offset_ = 0; }

 private:
  size_t offset_{0};
  char data_[0];
};

template <typename T>
class Iterator : IntermediateResult<T> {
 public:
  typedef std::function<void(page_id_t)> ReleasePageCallback;

 public:
  explicit Iterator(std::vector<page_id_t> pages, BufferPoolManager *bpm,
                    ReleasePageCallback release_page_callback = nullptr)
      : release_page_callback_(release_page_callback), bpm_(bpm), pages_(pages.begin(), pages.end()){};

  /**
   * Advance the iterator to the next tuple. If the current sort page is exhausted, move to the
   * next sort page.
   */
  auto operator++() -> Iterator & {
    BUSTUB_ENSURE(!End(), "Iterator has reached the end");

    if (++offset_ < tuples_in_current_page_.size()) {
      return *this;
    }

    cur_page_id_ = std::nullopt;
    tuples_in_current_page_.clear();
    offset_ = 0;

    if (pages_.empty()) {
      return *this;
    }

    cur_page_id_ = pages_.front();
    pages_.pop_front();

    auto guard = bpm_->ReadPage(cur_page_id_.value());
    auto page = guard.As<IntermediateResultPage<T>>();

    page->ReadAll(tuples_in_current_page_);
    BUSTUB_ENSURE(tuples_in_current_page_.size() > 0, "Page should contain at least one tuple");

    if (release_page_callback_) {
      release_page_callback_(cur_page_id_.value());
    }
    return *this;
  };

  /**
   * Dereference the iterator to get the current tuple in the sorted run that the iterator is
   * pointing to.
   */
  auto operator*() -> T {
    BUSTUB_ENSURE(!End(), "Iterator has reached the end");

    if (cur_page_id_ == std::nullopt) {
      ++(*this);
    }

    BUSTUB_ENSURE(offset_ < tuples_in_current_page_.size(), "Iterator out of bounds");
    return tuples_in_current_page_[offset_];
  };

  auto End() -> bool { return cur_page_id_ == std::nullopt && pages_.empty(); }

  auto Offset() const -> size_t { return offset_; }

 private:
  ReleasePageCallback release_page_callback_;
  BufferPoolManager *bpm_;

  /** The sorted run that the iterator is iterating on. */
  std::list<page_id_t> pages_;

  std::optional<page_id_t> cur_page_id_{std::nullopt};
  std::vector<Tuple> tuples_in_current_page_;
  size_t offset_{0};
};

}  // namespace bustub
