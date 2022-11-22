/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <algorithm>
#include <glog/logging.h>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/checks.h"
#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/loc.h"
#include "../cc/power.h"
#include "../cc/util.h"

#define MAX_SEQ_LEN 17
#define EOS_IDX -1

namespace dipcc {

// forward declares
std::vector<std::string> get_compound_build_orders(
    const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
        &all_possible_orders,
    std::vector<Loc> orderable_locs, int n_builds);

class ValidOrdersEncoder {
public:
  ValidOrdersEncoder(
      std::unordered_map<std::string, int> order_vocabulary_to_idx,
      int max_cands)
      : max_cands_(max_cands),
        order_vocabulary_to_idx_(order_vocabulary_to_idx){};

  py::tuple encode_valid_orders(const std::string &power, GameState &state);

  // pybind
  py::tuple encode_valid_orders_from_game(const std::string &power,
                                          Game &game) {
    return encode_valid_orders(power, game.get_state());
  }

private:
  // Methods
  int smarter_order_index(const Order &) const;
  std::vector<int> filter_orders_in_vocab(const std::set<Order> &) const;
  std::vector<Loc> get_sorted_actual_orderable_locs(
      const std::unordered_set<Loc> &root_locs,
      const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
          &all_possible_orders) const;

  // Data
  std::unordered_map<std::string, int> order_vocabulary_to_idx_;
  int max_cands_;
};

py::tuple ValidOrdersEncoder::encode_valid_orders(const std::string &power_s,
                                                  GameState &state) {
  Power power(power_from_str(power_s));

  // Init return value: all_order_idxs
  py::array_t<int32_t> all_order_idxs({1, MAX_SEQ_LEN, max_cands_});
  memset(all_order_idxs.mutable_data(0, 0, 0), EOS_IDX,
         MAX_SEQ_LEN * max_cands_ * sizeof(int32_t));

  // Init return value: loc_idxs
  py::array_t<int8_t> loc_idxs({1, 81});
  memset(loc_idxs.mutable_data(0, 0), -1, 81 * sizeof(int8_t));

  // Early exit?
  auto orderable_locs_it = state.get_orderable_locations().find(power);
  if (orderable_locs_it == state.get_orderable_locations().end() ||
      orderable_locs_it->second.size() == 0) {
    return py::make_tuple(all_order_idxs, loc_idxs, 0);
  }

  // Get orderable_locs sorted by coast-specific loc idx (orderable_locs returns
  // root_locs)
  auto &all_possible_orders(state.get_all_possible_orders());
  std::vector<Loc> orderable_locs(get_sorted_actual_orderable_locs(
      orderable_locs_it->second, all_possible_orders));

  int n_builds = state.get_n_builds(power);
  if (n_builds > 0) {
    // builds phase
    n_builds = std::min(n_builds, static_cast<int>(orderable_locs.size()));
    std::vector<std::string> orders(get_compound_build_orders(
        all_possible_orders, orderable_locs, n_builds));
    std::vector<int> order_idxs(orders.size());
    for (int j = 0; j < orders.size(); ++j) {
      order_idxs[j] = order_vocabulary_to_idx_.at(orders[j]);
    }
    std::sort(order_idxs.begin(), order_idxs.end());
    for (int j = 0; j < orders.size(); ++j) {
      *all_order_idxs.mutable_data(0, 0, j) = order_idxs[j];
    }
    for (Loc loc : orderable_locs) {
      *loc_idxs.mutable_data(0, static_cast<int>(root_loc(loc)) - 1) = -2;
    }
    return py::make_tuple(all_order_idxs, loc_idxs, n_builds);

  } else if (n_builds < 0) {
    // disband phase
    int n_disbands = -n_builds;
    std::vector<int> order_idxs;
    order_idxs.reserve(orderable_locs.size());
    for (Loc loc : orderable_locs) {
      for (int idx : filter_orders_in_vocab(all_possible_orders.at(loc))) {
        order_idxs.push_back(idx);
      }
    }
    std::sort(order_idxs.begin(), order_idxs.end());
    for (int i = 0; i < n_disbands; ++i) {
      for (int j = 0; j < order_idxs.size(); ++j) {
        *all_order_idxs.mutable_data(0, i, j) = order_idxs[j];
      }
    }
    for (Loc loc : orderable_locs) {
      *loc_idxs.mutable_data(0, static_cast<int>(root_loc(loc)) - 1) = -2;
    }
    return py::make_tuple(all_order_idxs, loc_idxs, n_disbands);

  } else {
    // move or retreat phase
    for (int i = 0; i < orderable_locs.size(); ++i) {
      Loc loc = orderable_locs[i];
      std::vector<int> order_idxs(
          filter_orders_in_vocab(all_possible_orders.at(loc)));
      std::sort(order_idxs.begin(), order_idxs.end());
      for (int j = 0; j < order_idxs.size(); ++j) {
        *all_order_idxs.mutable_data(0, i, j) = order_idxs[j];
        *loc_idxs.mutable_data(0, static_cast<int>(root_loc(loc)) - 1) = i;
      }
    }
    return py::make_tuple(all_order_idxs, loc_idxs, orderable_locs.size());
  }
} // encode_valid_orders

std::vector<int> ValidOrdersEncoder::filter_orders_in_vocab(
    const std::set<Order> &orders) const {
  std::vector<int> idxs;
  idxs.reserve(orders.size());

  for (const Order &order : orders) {
    int idx = smarter_order_index(order);
    if (idx != -1) {
      idxs.push_back(idx);
    }
  }

  return idxs;
}

int ValidOrdersEncoder::smarter_order_index(const Order &order) const {
  std::string order_s(order.to_string());
  auto it = order_vocabulary_to_idx_.find(order_s);
  if (it != order_vocabulary_to_idx_.end()) {
    return it->second;
  }

  // Try order with no coasts
  std::string order_s_no_coasts;
  order_s_no_coasts.reserve(order_s.size());
  for (int i = 0; i < order_s.size();) {
    char c = order_s[i];
    if (c == '/') {
      i += 3; // skip coast
    } else {
      order_s_no_coasts += c;
      i += 1;
    }
  }

  it = order_vocabulary_to_idx_.find(order_s_no_coasts);
  if (it != order_vocabulary_to_idx_.end()) {
    return it->second;
  }

  // Give up
  return -1;
}

std::vector<Loc> ValidOrdersEncoder::get_sorted_actual_orderable_locs(
    const std::unordered_set<Loc> &root_locs,
    const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
        &all_possible_orders) const {
  std::vector<Loc> locs;
  locs.reserve(root_locs.size());

  for (Loc rloc : root_locs) {
    auto &coasts = expand_coasts(rloc);
    if (coasts.size() == 1) {
      locs.push_back(rloc);
    } else {
      for (Loc cloc : coasts) {
        auto it = all_possible_orders.find(cloc);
        if (it != all_possible_orders.end() && it->second.size() > 0) {
          locs.push_back(cloc);
          break;
        }
      }
    }
  }

  std::sort(locs.begin(), locs.end());
  return locs;
}

// See combinations()
void combinations_impl(int min, int n, int c, std::vector<int> &v,
                       std::function<void(const std::vector<int> &)> foo) {
  for (int i = min; i <= (n - c); ++i) {
    v.push_back(i);
    if (c == 1) {
      foo(v);
    } else {
      combinations_impl(i + 1, n, c - 1, v, foo);
    }
    v.pop_back();
  }
}

// Call foo() once with each unique c-len combination of integers in [0, n-1]
void combinations(int n, int c,
                  std::function<void(const std::vector<int> &)> foo) {
  JCHECK(n >= c, "Called combinations with n < c");
  std::vector<int> v;
  v.reserve(c);
  combinations_impl(0, n, c, v, foo);
}

std::vector<std::string> get_compound_build_orders(
    const std::unordered_map<dipcc::Loc, std::set<dipcc::Order>>
        &all_possible_orders,
    std::vector<Loc> orderable_locs, int n_builds) {

  std::vector<std::string> r;
  r.reserve(64);

  combinations(orderable_locs.size(), n_builds,
               [&](const std::vector<int> &orderable_locs_idxs) {
                 std::vector<std::vector<Order>> order_lists;
                 order_lists.resize(n_builds);
                 int product = 1;

                 for (int i = 0; i < n_builds; ++i) {
                   for (const Order &order : all_possible_orders.at(
                            orderable_locs[orderable_locs_idxs[i]])) {
                     order_lists[i].push_back(order);
                   }
                   product *= order_lists[i].size();
                 }

                 std::vector<int> counter(n_builds, 0);

                 std::vector<std::string> orders_to_cat;
                 orders_to_cat.resize(n_builds);

                 for (int i = 0; i < product; ++i) {

                   // Gather orders to cat
                   for (int j = 0; j < n_builds; ++j) {
                     orders_to_cat[j] =
                         (order_lists[j][counter[j]].to_string());
                   }
                   std::sort(orders_to_cat.begin(), orders_to_cat.end());

                   // Cat orders and add to final list
                   std::string cat(orders_to_cat[0]);
                   for (int k = 1; k < orders_to_cat.size(); ++k) {
                     cat += ";";
                     cat += orders_to_cat[k];
                   }
                   r.push_back(cat);

                   // Increase counter
                   for (int j = n_builds - 1; j >= 0; --j) {
                     if (counter[j] == order_lists[j].size() - 1) {
                       counter[j] = 0;
                     } else {
                       counter[j] += 1;
                       break;
                     }
                   }
                 }
               });

  return r;
}

} // namespace dipcc
