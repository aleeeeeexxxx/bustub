// :bustub-keep-private:
//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// arc_replacer.cpp
//
// Identification: src/buffer/arc_replacer.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/arc_replacer.h"
#include "common/config.h"

namespace bustub {

/**
 *
 * TODO(P1): Add implementation
 *
 * @brief a new ArcReplacer, with lists initialized to be empty and target size to 0
 * @param num_frames the maximum number of frames the ArcReplacer will be required to cache
 */
ArcReplacer::ArcReplacer(size_t num_frames) : replacer_size_(num_frames) {}

/**
 * TODO(P1): Add implementation
 *
 * @brief Performs the Replace operation as described by the writeup
 * that evicts from either mfu_ or mru_ into its corresponding ghost list
 * according to balancing policy.
 *
 * If you wish to refer to the original ARC paper, please note that there are
 * two changes in our implementation:
 * 1. When the size of mru_ equals the target size, we don't check
 * the last access as the paper did when deciding which list to evict from.
 * This is fine since the original decision is stated to be arbitrary.
 * 2. Entries that are not evictable are skipped. If all entries from the desired side
 * (mru_ / mfu_) are pinned, we instead try victimize the other side (mfu_ / mru_),
 * and move it to its corresponding ghost list (mfu_ghost_ / mru_ghost_).
 *
 * @return frame id of the evicted frame, or std::nullopt if cannot evict
 */
auto ArcReplacer::Evict() -> std::optional<frame_id_t> {
  std::lock_guard<std::mutex> guard(latch_);

  if (curr_size_ == 0) {
    return std::nullopt;
  }

  curr_size_--;

  std::vector<std::reference_wrapper<std::list<frame_id_t>>> to_search;
  if (mru_.size() >= mru_target_size_) {
    to_search = {std::ref(mru_), std::ref(mfu_)};
  } else {
    to_search = {std::ref(mfu_), std::ref(mru_)};
  }

  std::shared_ptr<FrameStatus> victim;
  for (auto &lst_ref : to_search) {
    std::list<frame_id_t> &list = lst_ref.get();
    for (auto it = list.rbegin(); it != list.rend(); it++) {
      auto itr = alive_map_.find(*it);
      if (itr == alive_map_.end()) {
        throw std::logic_error("ArcReplacer::Evict(): alive_map_ inconsistent with lists");
      }
      auto frame = itr->second;
      if (frame->evictable_) {
        victim = frame;
        list.erase(std::next(it).base());
        break;
      }
    }
    if (victim != nullptr) {
      break;
    }
  }

  alive_map_.erase(victim->frame_id_);
  if (victim->arc_status_ == ArcStatus::MRU) {
    victim->arc_status_ = ArcStatus::MRU_GHOST;
    mru_ghost_.push_front(victim->page_id_);
  } else {
    victim->arc_status_ = ArcStatus::MFU_GHOST;
    mfu_ghost_.push_front(victim->page_id_);
  }
  ghost_map_[victim->page_id_] = victim;

  return victim->frame_id_;
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Record access to a frame, adjusting ARC bookkeeping accordingly
 * by bring the accessed page to the front of mfu_ if it exists in any of the lists
 * or the front of mru_ if it does not.
 *
 * Performs the operations EXCEPT REPLACE described in original paper, which is
 * handled by `Evict()`.
 *
 * Consider the following four cases, handle accordingly:
 * 1. Access hits mru_ or mfu_
 * 2/3. Access hits mru_ghost_ / mfu_ghost_
 * 4. Access misses all the lists
 *
 * This routine performs all changes to the four lists as preperation
 * for `Evict()` to simply find and evict a victim into ghost lists.
 *
 * Note that frame_id is used as identifier for alive pages and
 * page_id is used as identifier for the ghost pages, since page_id is
 * the unique identifier to the page after it's dead.
 * Using page_id for alive pages should be the same since it's one to one mapping,
 * but using frame_id is slightly more intuitive.
 *
 * @param frame_id id of frame that received a new access.
 * @param page_id id of page that is mapped to the frame.
 * @param access_type type of access that was received. This parameter is only needed for
 * leaderboard tests.
 */
void ArcReplacer::RecordAccess(frame_id_t frame_id, page_id_t page_id, [[maybe_unused]] AccessType access_type) {
  std::lock_guard<std::mutex> guard(latch_);

  auto alive_itr = alive_map_.find(frame_id);
  if (alive_itr != alive_map_.end()) {
    auto frame = alive_itr->second;
    assert(frame->page_id_ == page_id);

    if (frame->arc_status_ == ArcStatus::MFU) {
      // case 1: hit mru_ , move to the front of mfu_
      mfu_.remove(frame_id);
    } else {
      // case 2: hit mfu_, move to the front of mfu_
      mru_.remove(frame_id);
    }

    frame->arc_status_ = ArcStatus::MFU;
    mfu_.push_front(frame_id);
    return;
  }

  auto ghost_itr = ghost_map_.find(page_id);
  if (ghost_itr != ghost_map_.end()) {
    auto frame = ghost_itr->second;

    if (frame->arc_status_ == ArcStatus::MRU_GHOST) {
      // case 3: hit mru_ghost_, move to the front of mfu_, increase target size
      if (mru_ghost_.size() >= mfu_ghost_.size()) {
        IncreateTargetSize(1);
      } else {
        IncreateTargetSize(mfu_ghost_.size() / mru_ghost_.size());
      }
      mru_ghost_.remove(page_id);
    } else {
      // case 4: hit mfu_ghost_, move to the front of mfu_, decrease target size
      if (mfu_ghost_.size() >= mru_ghost_.size()) {
        IncreateTargetSize(-1);
      } else {
        IncreateTargetSize(-1 * mru_ghost_.size() / mfu_ghost_.size());
      }
      mfu_ghost_.remove(page_id);
    }

    frame->frame_id_ = frame_id;
    frame->page_id_ = page_id;
    frame->evictable_ = false;
    frame->arc_status_ = ArcStatus::MFU;

    mfu_.push_front(frame_id);

    alive_map_[frame_id] = frame;
    ghost_map_.erase(page_id);
    return;
  }

  // case 5: miss all lists, add to the front of mru_
  assert(mru_.size() + mru_ghost_.size() <= replacer_size_);

  auto select_ghost = [&]() -> std::optional<std::reference_wrapper<std::list<page_id_t>>> {
    if (mru_.size() + mru_ghost_.size() == replacer_size_) {
      return mru_ghost_;
    }
    if (mru_.size() + mru_ghost_.size() + mfu_.size() + mfu_ghost_.size() >= 2 * replacer_size_) {
      return mfu_ghost_;
    }
    return std::nullopt;
  };

  if (auto ghost = select_ghost(); ghost) {
    auto &victim = ghost->get();
    if (victim.empty()) {
      return;
    }
    auto to_delete = victim.back();
    victim.pop_back();
    ghost_map_.erase(to_delete);
  }

  mru_.push_front(frame_id);
  alive_map_[frame_id] = std::make_shared<FrameStatus>(page_id, frame_id, false, ArcStatus::MRU);
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Toggle whether a frame is evictable or non-evictable. This function also
 * controls replacer's size. Note that size is equal to number of evictable entries.
 *
 * If a frame was previously evictable and is to be set to non-evictable, then size should
 * decrement. If a frame was previously non-evictable and is to be set to evictable,
 * then size should increment.
 *
 * If frame id is invalid, throw an exception or abort the process.
 *
 * For other scenarios, this function should terminate without modifying anything.
 *
 * @param frame_id id of frame whose 'evictable' status will be modified
 * @param set_evictable whether the given frame is evictable or not
 */
void ArcReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::lock_guard<std::mutex> guard(latch_);

  auto frame = alive_map_.find(frame_id);
  if (frame == alive_map_.end()) {
    throw std::runtime_error("SetEvictable: frame_id not found");
  }

  if (frame->second->evictable_ != set_evictable) {
    frame->second->evictable_ = set_evictable;
    if (set_evictable) {
      curr_size_++;
    } else {
      curr_size_--;
    }
  }
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Remove an evictable frame from replacer.
 * This function should also decrement replacer's size if removal is successful.
 *
 * Note that this is different from evicting a frame, which always remove the frame
 * decided by the ARC algorithm.
 *
 * If Remove is called on a non-evictable frame, throw an exception or abort the
 * process.
 *
 * If specified frame is not found, directly return from this function.
 *
 * @param frame_id id of frame to be removed
 */
void ArcReplacer::Remove(frame_id_t frame_id) {}

/**
 * TODO(P1): Add implementation
 *
 * @brief Return replacer's size, which tracks the number of evictable frames.
 *
 * @return size_t
 */
auto ArcReplacer::Size() -> size_t {
  std::lock_guard<std::mutex> guard(latch_);
  return curr_size_;
}

}  // namespace bustub
