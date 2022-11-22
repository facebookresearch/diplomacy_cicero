/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace postman {
struct QueueClosed : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// TODO: Consider using atomics for size?
// TODO: Consider re-adding the timeouts?
template <typename T>
class Queue {
 public:
  Queue(int64_t max_size) : max_size_(max_size) {}

  int64_t size() const {
    std::unique_lock<std::mutex> lock(mu_);
    return deque_.size();
  }

  void enqueue(T item) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      while (!closed_ && deque_.size() >= max_size_) {
        can_dequeue_.wait(lock);
      }
      if (closed_) {
        throw QueueClosed("Enqueue to closed queue");
      }

      deque_.push_back(std::move(item));
    }

    can_dequeue_.notify_one();
  }

  T dequeue() {
    T item = [&]() {
      std::unique_lock<std::mutex> lock(mu_);
      while (!closed_ && deque_.empty()) {
        can_dequeue_.wait(lock);
      }

      if (closed_) throw QueueClosed("Dequeue from closed queue");

      T item = std::move(deque_.front());
      deque_.pop_front();
      return item;
    }();
    can_enqueue_.notify_one();
    return item;
  }

  bool is_closed() const {
    // TODO: Consider using atomic_bool closed_ and don't acquire lock here?
    std::unique_lock<std::mutex> lock(mu_);
    return closed_;
  }

  void close() {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (closed_) {
        throw QueueClosed("Queue was closed already");
      }
      closed_ = true;
      deque_.clear();
    }
    can_dequeue_.notify_all();
    can_enqueue_.notify_all();
  }

 private:
  mutable std::mutex mu_;

  const uint64_t max_size_;

  std::condition_variable can_dequeue_;
  std::condition_variable can_enqueue_;

  bool closed_ = false /* GUARDED_BY(mu_) */;
  std::deque<T> deque_ /* GUARDED_BY(mu_) */;
};

}  // namespace postman
