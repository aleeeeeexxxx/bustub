//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// page_guard.h
//
// Identification: src/include/storage/page/page_guard.h
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

#include "buffer/arc_replacer.h"
#include "buffer/buffer_pool_manager.h"
#include "storage/disk/disk_scheduler.h"
#include "storage/page/page.h"

namespace bustub {

class BufferPoolManager;
class FrameHeader;

class PageGuard {
 public:
  PageGuard() = default;
  PageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame, std::shared_ptr<ArcReplacer> replacer,
            std::shared_ptr<std::mutex> bpm_latch, std::shared_ptr<DiskScheduler> disk_scheduler, bool shared);
  PageGuard(const PageGuard &) = delete;
  auto operator=(const PageGuard &) -> PageGuard & = delete;
  PageGuard(PageGuard &&that) noexcept;
  auto operator=(PageGuard &&that) noexcept -> PageGuard &;

  ~PageGuard();

  auto GetPageId() const -> page_id_t;

  auto GetData() const -> const char *;
  template <class T>
  auto As() const -> const T * {
    return reinterpret_cast<const T *>(GetData());
  }

  auto GetDataMut() -> char *;
  template <class T>
  auto AsMut() -> T * {
    return reinterpret_cast<T *>(GetDataMut());
  }

  auto IsDirty() const -> bool;
  void Flush();
  void Drop();

  void MoveFrom(PageGuard &&that);

 private:
  bool shared_;

 protected:
  page_id_t page_id_;
  std::shared_ptr<FrameHeader> frame_;
  std::shared_ptr<ArcReplacer> replacer_;
  std::shared_ptr<std::mutex> bpm_latch_;
  std::shared_ptr<DiskScheduler> disk_scheduler_;
  bool is_valid_{false};
};

/**
 * @brief An RAII object that grants thread-safe read access to a page of data.
 *
 * The _only_ way that the BusTub system should interact with the buffer pool's page data is via page guards. Since
 * `ReadPageGuard` is an RAII object, the system never has to manually lock and unlock a page's latch.
 *
 * With `ReadPageGuard`s, there can be multiple threads that share read access to a page's data. However, the existence
 * of any `ReadPageGuard` on a page implies that no thread can be mutating the page's data.
 */
class ReadPageGuard : public PageGuard {
  /** @brief Only the buffer pool manager is allowed to construct a valid `ReadPageGuard.` */
  friend class BufferPoolManager;

 public:
  ReadPageGuard() = default;

 private:
  /** @brief Only the buffer pool manager is allowed to construct a valid `ReadPageGuard.` */
  explicit ReadPageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame, std::shared_ptr<ArcReplacer> replacer,
                         std::shared_ptr<std::mutex> bpm_latch, std::shared_ptr<DiskScheduler> disk_scheduler);
};

/**
 * @brief An RAII object that grants thread-safe write access to a page of data.
 *
 * The _only_ way that the BusTub system should interact with the buffer pool's page data is via page guards. Since
 * `WritePageGuard` is an RAII object, the system never has to manually lock and unlock a page's latch.
 *
 * With a `WritePageGuard`, there can be only be one thread that has exclusive ownership over the page's data. This
 * means that the owner of the `WritePageGuard` can mutate the page's data as much as they want. However, the existence
 * of a `WritePageGuard` implies that no other `WritePageGuard` or any `ReadPageGuard`s for the same page can exist at
 * the same time.
 */
class WritePageGuard : public PageGuard {
  /** @brief Only the buffer pool manager is allowed to construct a valid `WritePageGuard.` */
  friend class BufferPoolManager;

 public:
  WritePageGuard() = default;

 private:
  /** @brief Only the buffer pool manager is allowed to construct a valid `WritePageGuard.` */
  explicit WritePageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame, std::shared_ptr<ArcReplacer> replacer,
                          std::shared_ptr<std::mutex> bpm_latch, std::shared_ptr<DiskScheduler> disk_scheduler);
};

}  // namespace bustub
