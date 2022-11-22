/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <array>
#include <chrono>
#include <random>
#include <thread>

#include <gtest/gtest.h>

#include "prioritized_replay.h"

using namespace buffer;
using namespace rela;

// void print_dim2(const char* s, torch::Tensor t) {
//   std::cout << s << " [";
//   for (int i = 0; i < t.dim(); ++i) {
//     if (i) std::cout << ",";
//     std::cout << t.size(i);
//   }
//   std::cout << "]\n";
// }

TensorDict buildData() {
  TensorDict data;

  data["observations/x_board_state"] =
      torch::zeros({128, 835}).to(torch::kLong);
  data["observations/x_build_numbers"] =
      torch::zeros({128, 7}).to(torch::kLong);
  data["observations/x_in_adj_phase"] = torch::zeros({128}).to(torch::kLong);
  data["observations/x_loc_idxs"] = torch::zeros({128, 7, 81}).to(torch::kLong);
  data["observations/x_possible_actions"] =
      torch::zeros({128, 7, 17, 469}).to(torch::kLong);
  data["observations/x_prev_orders"] =
      torch::zeros({128, 2, 100}).to(torch::kLong);
  data["observations/x_prev_state"] = torch::zeros({128, 835}).to(torch::kLong);
  data["observations/x_season"] = torch::zeros({128, 3}).to(torch::kLong);
  data["done"] = torch::zeros({128});
  data["rewards"] = torch::zeros({128, 7});
  return data;
}

TEST(RelaTest, TestAddAndSample) {
  const int capacity = 100;
  NestPrioritizedReplay replay(capacity, 1, 0.1, 0.1, 1);

  for (int i = 0; i < 10; ++i) {
    std::cout << "Add " << i << std::endl;
    replay.add_one(buildData(), 1.0);
  }

  auto [batch, _] = replay.sample(10);
  auto rewards = batch.at("done");
  ASSERT_EQ(rewards.dim(), 2);
  // [time, batch]
  ASSERT_EQ(rewards.size(0), 128);
  ASSERT_EQ(rewards.size(1), 10);
}

TEST(RelaTest, TestAddAndSampleShuffled) {
  const int capacity = 100;
  NestPrioritizedReplay replay(capacity, 1, 0.1, 0.1, 1, /*shuffle=*/true);

  for (int i = 0; i < 10; ++i) {
    std::cout << "Add " << i << std::endl;
    replay.add_one(buildData(), 1.0);
  }

  auto [batch, _] = replay.sample(10);
  auto rewards = batch.at("done");
  ASSERT_EQ(rewards.dim(), 2);
  // [time, batch] = [1, time * batch].
  ASSERT_EQ(rewards.size(0), 1);
  ASSERT_EQ(rewards.size(1), 10 * 128);
}

TEST(RelaTest, TestNumel) {
  const int capacity = 5;
  NestPrioritizedReplay replay(capacity, 1, 0.1, 0.1, 1);

  int numel = 0;
  int first_size = -1;
  for (int i = 0; i < capacity + 1; ++i) {
    auto data = buildData();
    tensor_dict::for_each(
        data, [&numel](const torch::Tensor &t) { numel += t.numel(); });
    if (i == 0)
      first_size = numel;
    replay.add_one(buildData(), 1.0);
    ASSERT_EQ(numel, replay.total_numel());
  }

  replay.sample(capacity);
  ASSERT_EQ(numel - first_size, replay.total_numel());
}

TEST(RelaTest, TestBytes) {
  const int capacity = 5;
  NestPrioritizedReplay replay(capacity, 1, 0.1, 0.1, 1);

  int bytes = 0;
  int first_size = -1;
  for (int i = 0; i < capacity + 1; ++i) {
    auto data = buildData();
    tensor_dict::for_each(data, [&bytes](const torch::Tensor &t) {
      bytes += t.numel() * t.element_size();
    });
    if (i == 0)
      first_size = bytes;
    replay.add_one(buildData(), 1.0);
    ASSERT_EQ(bytes, replay.total_bytes()) << "i=" << i;
  }

  replay.sample(capacity);
  ASSERT_EQ(bytes - first_size, replay.total_bytes());
}
