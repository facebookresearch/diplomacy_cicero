/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
// Copyright 2019 The SEED Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copied from
 *   https://raw.githubusercontent.com/google-research/seed_rl/master/grpc/ops/grpc.cc
 * and modified. */
#pragma once

#include <future>

#include <ATen/ATen.h>

#include "blocking_counter.h"
#include "queue.h"

#include <nest.h>

typedef nest::Nest<at::Tensor> TensorNest;

namespace postman {
class ComputationQueue {
 public:
  struct Computation {
    // Represents one batched computation.
    Computation(uint32_t batch_size)
        : num_ready(batch_size),
          promise(),
          future(promise.get_future()),
          batch_size(batch_size) {}

    TensorNest get_inputs() { return std::move(inputs); }

    void set_outputs(TensorNest outputs) {
      promise.set_value(std::move(outputs));
    }

    void set_exception(std::exception_ptr e) { promise.set_exception(e); }

    TensorNest inputs;
    BlockingCounter num_ready;

    std::promise<TensorNest> promise;
    std::shared_future<TensorNest> future;

    uint32_t size{0};  // guarded by ComputationQueue::computation_mu_
    const uint32_t batch_size;
  };

  ComputationQueue(uint32_t batch_size)
      : batch_size_(batch_size), queue_(1024) {}

  std::shared_future<TensorNest> compute(const TensorNest& args,
                                         int64_t* index) {
    std::shared_ptr<Computation> computation;
    {
      std::unique_lock lock(computation_mu_);

      if (current_computation_ == nullptr) {
        current_computation_ = std::make_shared<Computation>(batch_size_);
        current_computation_->inputs =
            args.map([batch_size = batch_size_](const at::Tensor& t) {
              c10::IntArrayRef sizes = t.sizes();
              std::vector<int64_t> shape = {batch_size};
              shape.insert(shape.end(), sizes.begin(), sizes.end());

              return at::empty(shape, t.dtype());
            });
        try {
          queue_.enqueue(current_computation_);
        } catch (const QueueClosed& e) {
          current_computation_.reset();
          throw;
        }
      }

      computation = current_computation_;
      *index = computation->size++;

      if (*index == computation->batch_size - 1) {
        current_computation_.reset();
      }
    }

    // Copy input tensors to the batched input tensors.
    TensorNest::for_each(
        [index](at::Tensor& input, const at::Tensor& arg) {
          input[*index] = arg;
        },
        computation->inputs, args);

    computation->num_ready.DecrementCount();
    return computation->future;
  }

  void close() {
    std::unique_lock lock(computation_mu_);
    current_computation_.reset();
    queue_.close();
  }

  bool closed() { return queue_.is_closed(); }

  std::shared_ptr<Computation> get(bool wait_till_full = false) {
    std::shared_ptr<Computation> computation = queue_.dequeue();
    if (!wait_till_full) {
      uint32_t size;
      {
        std::unique_lock lock(computation_mu_);
        size = computation->size;
        if (size != computation->batch_size) {
          // computation is the current_computation_.
          current_computation_.reset();
        }
      }

      if (size < computation->batch_size) {
        computation->num_ready.DecrementCount(computation->batch_size - size);

        computation->inputs.for_each([size](at::Tensor& t) {
          c10::IntArrayRef sizes = t.sizes();
          std::vector<int64_t> shape(sizes.begin(), sizes.end());
          shape[0] = size;
          t.resize_(shape);
        });
      }
    }

    computation->num_ready.Wait();
    return computation;
  }

  void set_batch_size(uint32_t batch_size) {
    std::unique_lock lock(computation_mu_);
    batch_size_ = batch_size;
  }

 private:
  uint32_t batch_size_;  // GUARDED_BY(computation_mu_);

  std::mutex computation_mu_;
  std::shared_ptr<Computation>
      current_computation_;  // GUARDED_BY(computation_mu_);

  Queue<std::shared_ptr<Computation>> queue_;
};

}  // namespace postman
