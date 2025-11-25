//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// page_guard.cpp
//
// Identification: src/storage/page/page_guard.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/page/page_guard.h"
#include <memory>
#include <utility>
#include "buffer/arc_replacer.h"
#include "common/logger.h"
#include "common/macros.h"

namespace bustub {
/**
 * @brief The only constructor for an RAII `PageGuard` that creates a valid guard.
 *
 * Note that only the buffer pool manager is allowed to call this constructor.
 *
 */
PageGuard::PageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame, std::shared_ptr<ArcReplacer> replacer,
                     std::shared_ptr<std::mutex> bpm_latch, std::shared_ptr<DiskScheduler> disk_scheduler, bool shared)
    : shared_(shared),
      page_id_(page_id),
      frame_(std::move(frame)),
      replacer_(std::move(replacer)),
      bpm_latch_(std::move(bpm_latch)),
      disk_scheduler_(std::move(disk_scheduler)) {
  LOG_DEBUG("Locking page %d , mode=%s, frame=%d.", page_id_, shared_ ? "shared" : "exclusive", frame_->frame_id_);
  if (shared_) {
    frame_->rwlatch_.lock_shared();
  } else {
    frame_->rwlatch_.lock();
  }
  LOG_DEBUG("Locked page %d , mode=%s, frame=%d.", page_id_, shared_ ? "shared" : "exclusive", frame_->frame_id_);
  is_valid_ = true;
}

/**
 * @brief The move constructor for `PageGuard`.
 *
 */
PageGuard::PageGuard(PageGuard &&that) noexcept { MoveFrom(std::move(that)); }

/**
 * @brief The move assignment operator for `PageGuard`.
 *
 */
auto PageGuard::operator=(PageGuard &&that) noexcept -> PageGuard & {
  MoveFrom(std::move(that));
  return *this;
}

void PageGuard::MoveFrom(PageGuard &&that) {
  if (this == &that) {
    return;
  }

  Drop();

  page_id_ = that.page_id_;
  frame_ = std::move(that.frame_);
  replacer_ = std::move(that.replacer_);
  bpm_latch_ = std::move(that.bpm_latch_);
  disk_scheduler_ = std::move(that.disk_scheduler_);
  is_valid_ = that.is_valid_;
  shared_ = that.shared_;

  that.is_valid_ = false;
}

/**
 * @brief Gets the page ID of the page this guard is protecting.
 */
auto PageGuard::GetPageId() const -> page_id_t {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return page_id_;
}

/**
 * @brief Gets a `const` pointer to the page of data this guard is protecting.
 */
auto PageGuard::GetData() const -> const char * {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->GetData();
}

/**
 * @brief Gets a mutable pointer to the page of data this guard is protecting.
 */
auto PageGuard::GetDataMut() -> char * {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->GetDataMut();
}

/**
 * @brief Returns whether the page is dirty (modified but not flushed to the disk).
 */
auto PageGuard::IsDirty() const -> bool {
  BUSTUB_ENSURE(is_valid_, "tried to use an invalid write guard");
  return frame_->is_dirty_;
}

/**
 * @brief Flushes this page's data safely to disk.
 */
void PageGuard::Flush() { UNIMPLEMENTED("TODO(P1): Add implementation."); }

/**
 * @brief Manually drops a valid `PageGuard`'s data. If this guard is invalid, this function does nothing.
 *
 * ### Implementation
 *
 * Make sure you don't double free! Also, think **very** **VERY** carefully about what resources you own and the order
 * in which you release those resources. If you get the ordering wrong, you will very likely fail one of the later
 * Gradescope tests. You may also want to take the buffer pool manager's latch in a very specific scenario...
 *
 * TODO(P1): Add implementation.
 */
void PageGuard::Drop() {
  if (!is_valid_) {
    return;
  }

  is_valid_ = false;

  if (shared_) {
    frame_->rwlatch_.unlock_shared();
  } else {
    frame_->rwlatch_.unlock();
  }

  LOG_DEBUG("Page %d unlocked, mode=%s, frame=%d.", page_id_, shared_ ? "shared" : "exclusive", frame_->frame_id_);

  {
    std::lock_guard<std::mutex> lock(*bpm_latch_);
    frame_->pin_count_--;
    if (frame_->pin_count_ == 0) {
      LOG_DEBUG("frame %d for page %d is now unpinned.", frame_->frame_id_, page_id_);
      replacer_->SetEvictable(frame_->frame_id_, true);
    }
  }
}

/** @brief The destructor for `PageGuard`. This destructor simply calls `Drop()`. */
PageGuard::~PageGuard() { Drop(); }

ReadPageGuard::ReadPageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame,
                             std::shared_ptr<ArcReplacer> replacer, std::shared_ptr<std::mutex> bpm_latch,
                             std::shared_ptr<DiskScheduler> disk_scheduler)
    : PageGuard(page_id, std::move(frame), std::move(replacer), std::move(bpm_latch), std::move(disk_scheduler), true) {
}

WritePageGuard::WritePageGuard(page_id_t page_id, std::shared_ptr<FrameHeader> frame,
                               std::shared_ptr<ArcReplacer> replacer, std::shared_ptr<std::mutex> bpm_latch,
                               std::shared_ptr<DiskScheduler> disk_scheduler)
    : PageGuard(page_id, std::move(frame), std::move(replacer), std::move(bpm_latch), std::move(disk_scheduler),
                false) {
  frame_->is_dirty_ = true;
}

}  // namespace bustub
