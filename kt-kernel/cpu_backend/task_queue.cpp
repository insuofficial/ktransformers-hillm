/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iostream>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {
// Spin this long before sleeping on the condition variable: a condvar wake costs 7-14 us (more under
// load), while decode waits ~0.15-0.26 ms for a task and submits one every ~0.8 ms (~3 ms across steps).
constexpr auto kSyncSpin = std::chrono::milliseconds(1);
constexpr auto kDispatchSpin = std::chrono::milliseconds(10);

// Spin until pred() or the window ends; true if pred() held.
template <class Pred>
bool spin_until(Pred pred, std::chrono::steady_clock::duration window) {
  const auto t0 = std::chrono::steady_clock::now();
  while (!pred()) {
    if (std::chrono::steady_clock::now() - t0 > window) return false;
#if defined(__x86_64__)
    _mm_pause();
#endif
  }
  return true;
}
}  // namespace

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  auto ready = [&] {
    return pending.load(std::memory_order_acquire) <= allow_n_pending
        || done.load(std::memory_order_acquire);
  };
  // After a successful spin the wait below returns at once; the lock is still needed to read
  // the task exception.
  spin_until(ready, kSyncSpin);
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, ready);
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
    } else {
      auto has_work = [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      };
      if (spin_until(has_work, kDispatchSpin)) continue;
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, has_work);
    }
  }
}
