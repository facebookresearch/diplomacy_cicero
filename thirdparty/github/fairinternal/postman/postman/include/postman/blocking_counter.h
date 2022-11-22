/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
// Copied from
// https://github.com/abseil/abseil-cpp/blob/12bc53e0318d80569270a5b26ccbc62b52022b89/absl/synchronization/blocking_counter.cc
//   and modified.
//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// blocking_counter.h
// -----------------------------------------------------------------------------

#pragma once

#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

// BlockingCounter
//
// This class allows a thread to block for a pre-specified number of actions.
// `BlockingCounter` maintains a single non-negative abstract integer "count"
// with an initial value `initial_count`. A thread can then call `Wait()` on
// this blocking counter to block until the specified number of events occur;
// worker threads then call 'DecrementCount()` on the counter upon completion of
// their work. Once the counter's internal "count" reaches zero, the blocked
// thread unblocks.
//
// A `BlockingCounter` requires the following:
//     - its `initial_count` is non-negative.
//     - the number of calls to `DecrementCount()` on it is at most
//       `initial_count`.
//     - `Wait()` is called at most once on it.
//
// Given the above requirements, a `BlockingCounter` provides the following
// guarantees:
//     - Once its internal "count" reaches zero, no legal action on the object
//       can further change the value of "count".
//     - When `Wait()` returns, it is legal to destroy the `BlockingCounter`.
//     - When `Wait()` returns, the number of calls to `DecrementCount()` on
//       this blocking counter exactly equals `initial_count`.
//
// Example:
//     BlockingCounter bcount(N);         // there are N items of work
//     ... Allow worker threads to start.
//     ... On completing each work item, workers do:
//     ... bcount.DecrementCount();      // an item of work has been completed
//
//     bcount.Wait();                    // wait for all work to be complete
//
class BlockingCounter {
 public:
  explicit BlockingCounter(int initial_count)
      : count_(initial_count), num_waiting_(0) {}

  BlockingCounter(const BlockingCounter&) = delete;
  BlockingCounter& operator=(const BlockingCounter&) = delete;

  // BlockingCounter::DecrementCount()
  //
  // Decrements the counter's "count" by one, and return "count == 0". This
  // function requires that "count != 0" when it is called.
  //
  // Memory ordering: For any threads X and Y, any action taken by X
  // before it calls `DecrementCount()` is visible to thread Y after
  // Y's call to `DecrementCount()`, provided Y's call returns `true`.
  bool DecrementCount(const uint delta = 1) {
    {
      std::unique_lock l(count_mutex_);
      count_ -= delta;
      if (count_ < 0) {
        throw std::runtime_error(
            "BlockingCounter::DecrementCount() called too many times.  count=" +
            std::to_string(count_));
      }
    }
    count_cond_.notify_all();
    return count_ == 0;
  }

  // BlockingCounter::Wait()
  //
  // Blocks until the counter reaches zero. This function may be called at
  // most once. On return, `DecrementCount()` will have been called
  // "initial_count" times and the blocking counter may be destroyed.
  //
  // Memory ordering: For any threads X and Y, any action taken by X
  // before X calls `DecrementCount()` is visible to Y after Y returns
  // from `Wait()`.
  void Wait() {
    std::unique_lock l(count_mutex_);
    assert(count_ >= 0 && "BlockingCounter underflow");

    // only one thread may call Wait(). To support more than one thread,
    // implement a counter num_to_exit, like in the Barrier class.
    assert(num_waiting_ == 0 && "multiple threads called Wait()");
    num_waiting_++;

    while (count_ != 0) {
      count_cond_.wait(l);
    }

    // At this point, We know that all threads executing DecrementCount have
    // released the lock, and so will not touch this object again.
    // Therefore, the thread calling this method is free to delete the object
    // after we return from this method.
  }

 private:
  int count_;
  int num_waiting_;

  std::condition_variable count_cond_;
  std::mutex count_mutex_;
};
