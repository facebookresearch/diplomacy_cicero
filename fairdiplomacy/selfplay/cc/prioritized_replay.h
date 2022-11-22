/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// Original implementation from https://github.com/facebookresearch/rela

// This class implements a prioritized replay buffer allowing concurrent
// accesses and writes. It also supports batch sampling based on computed
// priority. The common accepted DataType is Nest, a general structure to handle
// torch Tensors. Paper link for DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
// https://openreview.net/pdf?id=H1Dy---0Z

#pragma once

#include <future>
#include <random>
#include <string>
#include <vector>

#include "serialization.h"
#include "tensor_dict.h"

using namespace rela;

namespace buffer {
template <class DataType> class ConcurrentQueue {
public:
  ConcurrentQueue(int capacity)
      : capacity(capacity), total_bytes_(0), total_numel_(0), head_(0),
        tail_(0), size_(0), safe_tail_(0), safe_size_(0), sum_(0),
        evicted_(capacity, false), elements_(capacity), weights_(capacity, 0) {}

  int safe_size(float *sum) const {
    std::unique_lock<std::mutex> lk(m_);
    if (sum != nullptr) {
      *sum = sum_;
    }
    return safe_size_;
  }

  int size() const {
    std::unique_lock<std::mutex> lk(m_);
    return size_;
  }

  void block_append(const std::vector<DataType> &block,
                    const torch::Tensor &weights) {
    int block_size = block.size();

    std::unique_lock<std::mutex> lk(m_);
    cv_size_.wait(lk, [=] { return size_ + block_size <= capacity; });

    int start = tail_;
    int end = (tail_ + block_size) % capacity;

    tail_ = end;
    size_ += block_size;
    check_size(head_, tail_, size_);

    lk.unlock();

    float sum = 0;
    auto weight_acc = weights.accessor<float, 1>();
    assert(weight_acc.size(0) == block_size);
    int64_t numel = 0;
    int64_t bytes = 0;
    for (int i = 0; i < block_size; ++i) {
      int j = (start + i) % capacity;
      elements_[j] = block[i];
      weights_[j] = weight_acc[i];
      sum += weight_acc[i];
      tensor_dict::for_each(elements_[j], [&numel](const torch::Tensor &t) {
        numel += t.numel();
      });
      tensor_dict::for_each(elements_[j], [&bytes](const torch::Tensor &t) {
        bytes += t.numel() * t.element_size();
      });
    }
    total_numel_ += numel;
    total_bytes_ += bytes;

    lk.lock();

    cv_tail_.wait(lk, [=] { return safe_tail_ == start; });
    safe_tail_ = end;
    safe_size_ += block_size;
    sum_ += sum;
    check_size(head_, safe_tail_, safe_size_);

    lk.unlock();
    cv_tail_.notify_all();
  }

  // ------------------------------------------------------------- //
  // block_pop, update are thread-safe against block_append
  // but they are NOT thread-safe against each other

  void block_pop(int block_size) {
    std::lock_guard<std::mutex> lk(pop_m_);
    double diff = 0;
    int head = head_;
    int64_t numel = 0;
    int64_t bytes = 0;
    for (int i = 0; i < block_size; ++i) {
      diff -= weights_[head];
      evicted_[head] = true;
      tensor_dict::for_each(elements_[head], [&numel](const torch::Tensor &t) {
        numel += t.numel();
      });
      tensor_dict::for_each(elements_[head], [&bytes](const torch::Tensor &t) {
        bytes += t.numel() * t.element_size();
      });
      head = (head + 1) % capacity;
    }
    total_numel_ -= numel;
    total_bytes_ -= bytes;

    {
      std::lock_guard<std::mutex> lk(m_);
      sum_ += diff;
      head_ = head;
      safe_size_ -= block_size;
      size_ -= block_size;
      assert(safe_size_ >= 0);
      check_size(head_, safe_tail_, safe_size_);
    }
    cv_size_.notify_all();
  }

  void update(const std::vector<int> &ids, const torch::Tensor &weights) {
    double diff = 0;
    auto weight_acc = weights.accessor<float, 1>();
    for (int i = 0; i < (int)ids.size(); ++i) {
      auto id = ids[i];
      if (evicted_[id]) {
        continue;
      }
      diff += (weight_acc[i] - weights_[id]);
      weights_[id] = weight_acc[i];
    }

    std::lock_guard<std::mutex> lk_(m_);
    sum_ += diff;
  }

  // ------------------------------------------------------------- //
  // accessing elements is never locked, operate safely!

  DataType get_element_and_mark(int idx) {
    int id = (head_ + idx) % capacity;
    evicted_[id] = false;
    return elements_[id];
  }

  DataType get_element(int idx) {
    int id = (head_ + idx) % capacity;
    return elements_[id];
  }

  float get_weight(int idx, int *id) {
    assert(id != nullptr);
    *id = (head_ + idx) % capacity;
    return weights_[*id];
  }

  void save(const std::string &fpath) {
    // making a full copy as most of the memory are in tensors, not maps.
    std::vector<DataType> elements_copy;
    {
      std::lock_guard<std::mutex> lk(pop_m_);
      int head = head_;
      const int size = safe_size_;
      elements_copy.resize(size);
      for (int i = 0; i < size; ++i) {
        elements_copy[i] = elements_[i];
        head = (head + 1) % capacity;
      }
    }

    FILE *stream = fopen(fpath.c_str(), "wb");
    write(elements_copy, stream);
    fclose(stream);
  }

  size_t load(const std::string &fpath) {
    FILE *stream = fopen(fpath.c_str(), "rb");
    const auto elements = read(stream);
    fclose(stream);
    const auto weights = torch::ones({(int)elements.size()});
    block_append(elements, weights);
    return elements.size();
  }

  const int capacity;
  std::atomic<int64_t> total_bytes_;
  std::atomic<int64_t> total_numel_;

private:
  void check_size(int head, int tail, int size) {
    if (size == 0) {
      assert(tail == head);
    } else if (tail > head) {
      if (tail - head != size) {
        std::cout << "tail-head: " << tail - head << " vs size: " << size
                  << std::endl;
      }
      assert(tail - head == size);
    } else {
      if (tail + capacity - head != size) {
        std::cout << "tail-head: " << tail + capacity - head
                  << " vs size: " << size << std::endl;
      }
      assert(tail + capacity - head == size);
    }
  }
  mutable std::mutex m_;
  mutable std::mutex pop_m_;
  std::condition_variable cv_size_;
  std::condition_variable cv_tail_;

  int head_;
  int tail_;
  int size_;

  int safe_tail_;
  int safe_size_;
  double sum_;
  std::vector<int> evicted_;

  std::vector<DataType> elements_;
  std::vector<float> weights_;
};

template <class DataType> class PrioritizedReplay {
public:
  PrioritizedReplay(int capacity, int seed, float alpha, float beta,
                    int prefetch, bool shuffle = false)
      : alpha_(alpha) // priority exponent
        ,
        beta_(beta) // importance sampling exponent
        ,
        prefetch_(prefetch), shuffle_cross_chunks_(shuffle),
        capacity_(capacity), storage_(int(1.25 * capacity)), num_add_(0) {
    rng_.seed(seed);
  }

  void add_compressed(const std::vector<DataType> &sample,
                      const torch::Tensor &priority) {
    assert(priority.dim() == 1);
    assert(priority.size(0) == (int)sample.size());
    auto weights = torch::pow(priority, alpha_);
    storage_.block_append(sample, weights);
    num_add_ += priority.size(0);
  }

  void add_one(const DataType &sample, float priority) {
    add_compressed({sample}, torch::tensor({priority}));
  }

  void save(const std::string &fpath) { storage_.save(fpath); }

  void load(const std::string &fpath) { storage_.load(fpath); }

  std::tuple<std::vector<DataType>, torch::Tensor> get_all_content() {
    std::vector<DataType> samples;
    const int sample_size = size();
    auto weights = torch::ones({sample_size}, torch::kFloat32);
    auto weight_acc = weights.accessor<float, 1>();
    int id = 0;
    if (sample_size == 0) {
      return std::make_tuple(samples, weights);
    }
    int cur = 0;
    while (cur < sample_size) {
      DataType element = storage_.get_element(cur);
      samples.push_back(element);
      weight_acc[cur] = storage_.get_weight(cur, &id);
      cur++;
    }

    return std::make_tuple(std::move(samples), weights);
  }

  std::tuple<int, std::vector<DataType>, torch::Tensor> get_new_content() {
    std::vector<DataType> samples;
    int sample_size = num_add_ - last_query_;
    auto weights = torch::ones({sample_size}, torch::kFloat32);
    auto weight_acc = weights.accessor<float, 1>();
    int id = 0;
    if (sample_size == 0) {
      return std::make_tuple(0, samples, weights);
    }
    int cur = 0;
    while (cur < sample_size) {
      DataType element = storage_.get_element_and_mark(cur);
      samples.push_back(element);
      weight_acc[cur] = storage_.get_weight(cur, &id);
      last_query_++;
      cur++;
    }
    storage_.block_pop(sample_size);

    return std::make_tuple(sample_size, std::move(samples), weights);
  }

  // assuming batch is a vector to be added to the replay buffer
  void add_batch(const std::vector<DataType> &vecs,
                 const torch::Tensor &priority) {
    for (size_t i = 0; i < vecs.size(); i++) {
      auto priority_accessor = priority.accessor<float, 1>();
      add_one(vecs[i], priority_accessor[i]);
    }
  }

  // assuming batch is a vector to be added to the replay buffer
  // async version
  std::future<void> add_batch_async(const std::vector<DataType> &batch,
                                    const torch::Tensor &priority) {
    auto fut = [=] { add_batch(batch, priority); };
    return std::async(std::launch::async, fut);
  }

  std::tuple<DataType, torch::Tensor> sample(int batchsize) {
    if (!sampled_ids_.empty()) {
      std::cout << "Error: previous samples' priority has not been updated."
                << std::endl;
      assert(false);
    }

    assert(size() >= batchsize &&
           "Cannot sample from a buffer that doesn't have even batch");

    DataType batch;
    torch::Tensor priority;
    if (prefetch_ == 0) {
      std::tie(batch, priority, sampled_ids_) = sample_(batchsize);
      return std::make_tuple(batch, priority);
    }

    if (futures_.empty()) {
      std::tie(batch, priority, sampled_ids_) = sample_(batchsize);
    } else {
      // assert(futures_.size() == 1);
      std::tie(batch, priority, sampled_ids_) = futures_.front().get();
      futures_.pop();
    }

    while ((int)futures_.size() < prefetch_) {
      auto f =
          std::async(std::launch::async, &PrioritizedReplay<DataType>::sample_,
                     this, batchsize);
      futures_.push(std::move(f));
    }

    return std::make_tuple(batch, priority);
  }

  void update_priority(const torch::Tensor &priority) {
    assert(priority.dim() == 1);
    assert((int)sampled_ids_.size() == priority.size(0));

    auto weights = torch::pow(priority, alpha_);
    {
      std::lock_guard<std::mutex> lk(m_sampler_);
      storage_.update(sampled_ids_, weights);
    }
    sampled_ids_.clear();
  }

  void keep_priority() { sampled_ids_.clear(); }

  int size() const { return storage_.safe_size(nullptr); }

  int num_add() const { return num_add_; }

  int64_t total_numel() const { return storage_.total_numel_; }
  int64_t total_bytes() const { return storage_.total_bytes_; }

private:
  using SampleWeightIds = std::tuple<DataType, torch::Tensor, std::vector<int>>;

  SampleWeightIds sample_(int batchsize) {
    return shuffle_cross_chunks_ ? sample_shuffled_(batchsize)
                                 : sample_chunk_(batchsize);
  }
  SampleWeightIds sample_chunk_(int batchsize) {
    std::unique_lock<std::mutex> lk(m_sampler_);

    float sum;
    int size = storage_.safe_size(&sum);
    // storage_ [0, size) remains static in the subsequent section

    float segment = sum / batchsize;
    std::uniform_real_distribution<float> dist(0.0, segment);

    std::vector<DataType> samples;
    auto weights = torch::zeros({batchsize}, torch::kFloat32);
    auto weight_acc = weights.accessor<float, 1>();
    std::vector<int> ids(batchsize);

    double acc_sum = 0;
    int next_idx = 0;
    float w = 0;
    int id = 0;
    for (int i = 0; i < batchsize; i++) {
      float rand = dist(rng_) + i * segment;
      rand = std::min(sum - (float)0.2, rand);

      while (next_idx <= size) {
        if (acc_sum > 0 && acc_sum >= rand) {
          assert(next_idx >= 1);
          DataType element = storage_.get_element_and_mark(next_idx - 1);
          samples.push_back(element);
          weight_acc[i] = w;
          ids[i] = id;
          break;
        }

        if (next_idx == size) {
          std::cout << "next_idx: " << next_idx << "/" << size << std::endl;
          std::cout << std::setprecision(10) << "acc_sum: " << acc_sum
                    << ", sum: " << sum << ", rand: " << rand << std::endl;
          assert(false);
        }

        w = storage_.get_weight(next_idx, &id);
        acc_sum += w;
        ++next_idx;
      }
    }
    assert((int)samples.size() == batchsize);

    // pop storage if full
    size = storage_.size();
    if (size > capacity_) {
      storage_.block_pop(size - capacity_);
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();

    weights = weights / sum;
    weights = torch::pow(size * weights, -beta_);
    weights /= weights.max();

    DataType batch = tensor_dict::stack(samples, 1);
    return std::make_tuple(batch, weights, ids);
  }

  SampleWeightIds sample_shuffled_(int batchsize) {
    std::unique_lock<std::mutex> lk(m_sampler_);

    float sum;
    const int size = storage_.safe_size(&sum);
    const int chunk_size = storage_.get_element(0).begin()->second.size(0);

    std::vector<DataType> samples;
    std::uniform_int_distribution<> dist_chunk(0, size - 1);
    std::uniform_int_distribution<> dist_position(0, chunk_size - 1);
    for (int i = 0; i < batchsize * chunk_size; ++i) {
      const DataType &chunk = storage_.get_element(dist_chunk(rng_));
      const int index_in_chunk = dist_position(rng_);
      samples.push_back(tensor_dict::index(chunk, index_in_chunk));
    }

    {
      // pop storage if full
      auto real_size = storage_.size();
      if (real_size > capacity_) {
        storage_.block_pop(real_size - capacity_);
      }
    }

    // safe to unlock, because <samples> contains copys
    lk.unlock();

    DataType batch = tensor_dict::stack(samples, 0);
    // Adding time dimension.
    batch = tensor_dict::apply(batch,
                               [](torch::Tensor t) { return t.unsqueeze(0); });
    // FIXME(akhti): get rid of priorities.
    std::vector<int> ids;
    auto weights = torch::zeros({batchsize * chunk_size}, torch::kFloat32);
    return std::make_tuple(batch, weights, ids);
  }

  const float alpha_;
  const float beta_;
  const int prefetch_;
  const int capacity_;
  const bool shuffle_cross_chunks_;

  ConcurrentQueue<DataType> storage_;
  std::atomic<int> num_add_;

  // make sure that sample & update does not overlap
  std::mutex m_sampler_;
  std::vector<int> sampled_ids_;
  std::queue<std::future<SampleWeightIds>> futures_;

  std::mt19937 rng_;
  int last_query_ = 0;
};

using NestPrioritizedReplay = PrioritizedReplay<TensorDict>;

} // namespace buffer
