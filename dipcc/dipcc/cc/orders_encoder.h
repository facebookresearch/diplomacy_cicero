/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <map>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "checks.h"
#include "game.h"
#include "game_state.h"
#include "loc.h"
#include "power.h"
#include "util.h"

namespace dipcc {

class OrdersEncoder {
public:
  const static int MAX_SEQ_LEN = 17;
  const static int EOS_IDX = -1;

  OrdersEncoder(
      const std::unordered_map<std::string, int> &order_vocabulary_to_idx,
      int max_cands, bool allow_buggy_duplicates);

  // Encode x_valid_orders and x_loc_idxs into pre-allocated memory pointed to
  // by r_order_idxs and r_loc_idxs. Return the sequence length.
  void encode_valid_orders(Power power, GameState &state, int32_t *r_order_idxs,
                           int8_t *r_loc_idxs) const;

  // Perform "all-powers" single-sequence encoding into the tensors pointed to
  // by r_*
  void encode_valid_orders_all_powers(GameState &state, int32_t *r_order_idxs,
                                      int8_t *r_loc_idxs,
                                      int64_t *r_powers) const;

  // Encode x_prev_orders into pre-allocated memory pointed to by r.
  void encode_prev_orders_deepmind(const Game *game, long *r) const;
  void encode_orders_deepmind(
      const std::vector<Order> &orders,
      const std::vector<const GameState *> &state_for_each_order,
      long *r) const;

  int get_max_cands() const { return max_cands_; }

private:
  // Methods
  int smarter_order_index(const Order &) const;
  std::vector<int> filter_orders_in_vocab(const std::set<Order> &) const;
  template <typename T>
  std::vector<Loc> get_sorted_actual_orderable_locs(
      const T &root_locs,
      const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
          &all_possible_orders) const;
  void encode_adj_phase(Power power, GameState &state, int32_t *r_order_idxs,
                        int8_t *r_loc_idxs) const;

  // Data
  std::unordered_map<std::string, int> order_vocabulary_to_idx_;
  int max_cands_;
  // If true, behave in a buggy fashion that sometimes outputs duplicate coastal
  // orders, preserving old behavior pre-mid-November 2021.
  bool allow_buggy_duplicates_;
};

class OrdersDecoder {
public:
  OrdersDecoder(
      const std::unordered_map<std::string, int> &order_vocabulary_to_idx);

  // Decode a [B, 7, S]-shape tensor of EOS_IDX-padded order idxs.
  // Returns a 3d vector of string (batch, power, orders)
  std::vector<std::vector<std::vector<std::string>>>
  decode_order_idxs(torch::Tensor *order_idxs) const;
  // Same for all-power outputs.
  std::vector<std::vector<std::vector<std::string>>>
  decode_order_idxs_all_powers(torch::Tensor *order_idxs,
                               torch::Tensor *x_in_adj_phase,
                               torch::Tensor *x_power,
                               int batch_repeat_interleave) const;

private:
  // Data
  std::vector<std::string> order_vocabulary_;
};

} // namespace dipcc
