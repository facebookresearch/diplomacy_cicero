/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "orders_encoder.h"
#include <algorithm>
#include <glog/logging.h>
#include <string>
#include <utility>
#include <vector>

#define P_IDX(r, w, i, j) (*((r) + ((i) * (w)) + (j)))

#define PREV_ORDERS_WIDTH 100

using namespace std;

namespace dipcc {

// forward declares
vector<string> get_compound_build_orders(
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders,
    vector<Loc> orderable_locs, int n_builds);

// Constructor
OrdersDecoder::OrdersDecoder(
    const std::unordered_map<std::string, int> &order_vocabulary_to_idx) {
  // init order_vocabulary_
  int max_idx = 0;
  for (auto &p : order_vocabulary_to_idx) {
    if (p.second > max_idx) {
      max_idx = p.second;
    }
  }
  order_vocabulary_.resize(max_idx + 1);
  for (auto &p : order_vocabulary_to_idx) {
    order_vocabulary_[p.second] = p.first;
  }
}

// Constructor
OrdersEncoder::OrdersEncoder(
    const std::unordered_map<std::string, int> &order_vocabulary_to_idx,
    int max_cands, bool allow_buggy_duplicates)
    : order_vocabulary_to_idx_(order_vocabulary_to_idx), max_cands_(max_cands),
      allow_buggy_duplicates_(allow_buggy_duplicates) {}

void OrdersEncoder::encode_prev_orders_deepmind(const Game *game,
                                                long *r) const {
  std::vector<Order> prev_orders;
  std::vector<const GameState *> state_for_each_order;
  prev_orders.reserve(100);
  state_for_each_order.reserve(100);

  JCHECK(game->get_order_history().size() == game->get_state_history().size());
  auto orderit = game->get_order_history().rbegin();
  auto stateit = game->get_state_history().rbegin();
  while (orderit != game->get_order_history().rend() &&
         stateit != game->get_state_history().rend()) {

    for (auto &jt : *(orderit->second)) {
      for (const Order &order : jt.second) {
        prev_orders.push_back(order);
        state_for_each_order.push_back(stateit->second.get());
      }
    }

    // Encode up to and including the most recent movement phase
    if (orderit->first.phase_type == 'M') {
      break;
    }
    ++orderit;
    ++stateit;
  }

  encode_orders_deepmind(prev_orders, state_for_each_order, r);
} // encode_prev_orders_deepmind

void OrdersEncoder::encode_orders_deepmind(
    const std::vector<Order> &orders,
    const std::vector<const GameState *> &state_for_each_order, long *r) const {
  memset(r, 0, 2 * PREV_ORDERS_WIDTH * sizeof(long));

  vector<pair<int32_t, int8_t>> orders_pairs;
  orders_pairs.reserve(100);

  assert(orders.size() == state_for_each_order.size());
  for (size_t i = 0; i < orders.size(); ++i) {
    const Order &order = orders[i];
    const GameState *state = state_for_each_order[i];
    auto x = order_vocabulary_to_idx_.find(order.to_string());
    if (x != order_vocabulary_to_idx_.end()) {
      int32_t order_idx = x->second;
      int8_t loc_idx = static_cast<int>(order.get_unit().loc) - 1;
      orders_pairs.push_back(make_pair(order_idx, loc_idx));
    } else {
      // If the order isn't found, then try again in case of a support hold/move
      // having an imprecise coast for the supportee. Our order vocabulary only
      // has coast-qualified supportee orders.
      if (state != nullptr && (order.get_type() == OrderType::SH ||
                               order.get_type() == OrderType::SM)) {
        OwnedUnit supportee = state->get_unit_rooted(order.get_target().loc);
        // The supportee must be on a special coast if it's a fleet and getting
        // it via root location gives back a unit whose actual location is not
        // the location it was looked up by.
        if (supportee.type == UnitType::FLEET &&
            supportee.loc != order.get_target().loc) {
          Order alternative_order(order.get_unit(), order.get_type(),
                                  supportee.unowned(), order.get_dest(),
                                  order.get_via());
          x = order_vocabulary_to_idx_.find(alternative_order.to_string());
          if (x != order_vocabulary_to_idx_.end()) {
            int32_t order_idx = x->second;
            int8_t loc_idx =
                static_cast<int>(alternative_order.get_unit().loc) - 1;
            orders_pairs.push_back(make_pair(order_idx, loc_idx));
          }
        }
      }
    }
  }

  JCHECK(orders_pairs.size() < PREV_ORDERS_WIDTH, "orders exceeds max size");

  // Sort for deterministic encoding
  sort(orders_pairs.begin(), orders_pairs.end());

  // Add to preallocated tensor
  for (int i = 0; i < orders_pairs.size(); ++i) {
    pair<int32_t, int8_t> p = orders_pairs[i];
    P_IDX(r, PREV_ORDERS_WIDTH, 0, i) = p.first;
    P_IDX(r, PREV_ORDERS_WIDTH, 1, i) = p.second;
  }
}

void OrdersEncoder::encode_valid_orders_all_powers(GameState &state,
                                                   int32_t *r_order_idxs,
                                                   int8_t *r_loc_idxs,
                                                   int64_t *r_powers) const {
  // r_order_idxs[batch,i,j,k]
  // On moves and retreat phases, only i=0 is used.
  // j indexes the jth orderable location in order, and k indexes the kth
  // possible global order idx at that location. On adjustment phases, i indexes
  // the ith power. For builds, only j = 0 is used, k indexes the kth possible
  // global order idx for combined builds. For disbands, j indexes the jth
  // disband, k indexes the kth possible global order idx for a one-unit
  // disband.
  //
  // r_loc_idxs[batch,loc]
  // loc indexes locations.
  // On moves and retreat phases, the value at loc is j if loc s the jtth
  // orderable location. On adjustment phases, loc is -2 on locations that can
  // build (for builds) or on units to disband (for disbanding)
  //
  // r_powers[batch,i,j]
  // On moves and retreat phases, only i=0 is used.and j indexes the jth
  // orderable location in order, and the value is the power that is acting. On
  // adjustment phases, i indexes the ith power, and j indexes the jth orderable
  // location in order, and the value is the power that is acting, which is
  // always i.

  // Init return values
  memset(r_order_idxs, EOS_IDX,
         7 * N_SCS * max_cands_ * sizeof(int32_t));       // [1, 7, 34, 469]
  memset(r_loc_idxs, EOS_IDX, 81 * sizeof(int8_t));       // [1, 81]
  memset(r_powers, EOS_IDX, 7 * N_SCS * sizeof(int64_t)); // [1, 7, 34]

  if (state.get_phase().phase_type == 'A') {
    // Encode adj phase separately for each power; note we index the return
    // values for each power
    for (int i = 0; i < 7; ++i) {
      encode_adj_phase(POWERS[i], state,
                       (r_order_idxs + (i * (N_SCS * max_cands_))),
                       (r_loc_idxs + (i * 81)));

      // For adj-phase, set all x_power[i,:] = i
      for (int j = 0; j < N_SCS; ++j) {
        P_IDX(r_powers, N_SCS, i, j) = i;
      }
    }
  } else {
    // Get all orderable locs by all powers
    vector<int8_t> loc_powers(81, EOS_IDX);
    unordered_set<Loc> all_orderable_locs;
    for (auto &[power, locs] : state.get_orderable_locations()) {
      for (Loc loc : locs) {
        loc_powers[static_cast<int>(loc) - 1] = static_cast<int8_t>(power) - 1;
        all_orderable_locs.insert(loc);
      }
    }

    // Get orderable_locs sorted by coast-specific loc idx (orderable_locs
    // returns root_locs)
    auto &all_possible_orders(state.get_all_possible_orders());
    vector<Loc> orderable_locs(get_sorted_actual_orderable_locs(
        all_orderable_locs, all_possible_orders));

    // Encode outputs at each step
    for (int step = 0; step < orderable_locs.size(); ++step) {
      Loc loc = orderable_locs[step];
      vector<int> order_idxs(
          filter_orders_in_vocab(all_possible_orders.at(loc)));
      sort(order_idxs.begin(), order_idxs.end());

      // Encode order idxs
      for (int j = 0; j < order_idxs.size(); ++j) {
        P_IDX(r_order_idxs, max_cands_, step, j) = order_idxs[j];
      }

      // Encode x_loc_idxs
      int rloc_idx = static_cast<int>(root_loc(loc)) - 1;
      r_loc_idxs[rloc_idx] = step;

      // Encode x_power
      auto power_idx = loc_powers[rloc_idx];
      JCHECK(power_idx >= 0, "Orderable loc with no unit? Something's wrong");
      r_powers[step] = power_idx;
    }
  }
} // encode_valid_orders_all_powers

void OrdersEncoder::encode_adj_phase(Power power, GameState &state,
                                     int32_t *r_order_idxs,
                                     int8_t *r_loc_idxs) const {
  auto &all_possible_orders(state.get_all_possible_orders());
  auto orderable_locs_it = state.get_orderable_locations().find(power);
  if (orderable_locs_it == state.get_orderable_locations().end() ||
      orderable_locs_it->second.size() == 0) {
    return;
  }
  vector<Loc> orderable_locs(get_sorted_actual_orderable_locs(
      orderable_locs_it->second, all_possible_orders));

  int n_builds = state.get_n_builds(power);
  if (n_builds > 0) {
    // builds phase
    n_builds = min(n_builds, static_cast<int>(orderable_locs.size()));
    vector<string> orders(get_compound_build_orders(all_possible_orders,
                                                    orderable_locs, n_builds));
    vector<int> order_idxs(orders.size());
    for (int j = 0; j < orders.size(); ++j) {
      order_idxs[j] = order_vocabulary_to_idx_.at(orders[j]);
    }
    sort(order_idxs.begin(), order_idxs.end());
    for (int j = 0; j < orders.size(); ++j) {
      P_IDX(r_order_idxs, max_cands_, 0, j) = order_idxs[j];
    }
    for (Loc loc : orderable_locs) {
      r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = -2;
    }
    return;

  } else if (n_builds < 0) {
    // disband phase
    int n_disbands = -n_builds;
    vector<int> order_idxs;
    order_idxs.reserve(orderable_locs.size());
    for (Loc loc : orderable_locs) {
      for (int idx : filter_orders_in_vocab(all_possible_orders.at(loc))) {
        order_idxs.push_back(idx);
      }
    }
    sort(order_idxs.begin(), order_idxs.end());
    for (int i = 0; i < n_disbands; ++i) {
      for (int j = 0; j < order_idxs.size(); ++j) {
        P_IDX(r_order_idxs, max_cands_, i, j) = order_idxs[j];
      }
    }
    for (Loc loc : orderable_locs) {
      r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = -2;
    }
    return;
  }
} // encode_adj_phase

void OrdersEncoder::encode_valid_orders(Power power, GameState &state,
                                        int32_t *r_order_idxs,
                                        int8_t *r_loc_idxs) const {
  // r_order_idxs[batch,i,j,k]
  // i indexes the ith power (each call to this function only handles one i),
  // On moves and retreat phases, j indexes the jth orderable location in order
  // for that power, k indexes the kth possible global order idx at that
  // location. On adjustment phases, For builds, only j = 0 is used, k indexes
  // the kth possible global order idx for combined builds. For disbands, j
  // indexes the jth disband, k indexes the kth possible global order idx for a
  // one-unit disband.
  //
  // r_loc_idxs[batch,loc]
  // loc indexes locations.
  // On moves and retreat phases, the value at loc is j if loc is the jth
  // orderable location for that power. On adjustment phases, loc is -2 on
  // locations that can build (for builds) or on units to disband (for
  // disbanding)

  // Init return value: all_order_idxs
  // py::array_t<int32_t> all_order_idxs({1, MAX_SEQ_LEN, max_cands_});
  memset(r_order_idxs, EOS_IDX, MAX_SEQ_LEN * max_cands_ * sizeof(int32_t));

  // Init return value: loc_idxs
  // py::array_t<int8_t> loc_idxs({1, 81});
  memset(r_loc_idxs, EOS_IDX, 81 * sizeof(int8_t));

  // Early exit?
  auto orderable_locs_it = state.get_orderable_locations().find(power);
  if (orderable_locs_it == state.get_orderable_locations().end() ||
      orderable_locs_it->second.size() == 0) {
    return;
  }

  // Get orderable_locs sorted by coast-specific loc idx (orderable_locs
  // returns root_locs)
  auto &all_possible_orders(state.get_all_possible_orders());
  vector<Loc> orderable_locs(get_sorted_actual_orderable_locs(
      orderable_locs_it->second, all_possible_orders));
  if (state.get_phase().phase_type == 'A') {
    // adj phase
    encode_adj_phase(power, state, r_order_idxs, r_loc_idxs);
    return;
  } else {
    // move or retreat phase
    for (int i = 0; i < orderable_locs.size(); ++i) {
      Loc loc = orderable_locs[i];
      vector<int> order_idxs(
          filter_orders_in_vocab(all_possible_orders.at(loc)));
      sort(order_idxs.begin(), order_idxs.end());
      for (int j = 0; j < order_idxs.size(); ++j) {
        P_IDX(r_order_idxs, max_cands_, i, j) = order_idxs[j];
        r_loc_idxs[static_cast<int>(root_loc(loc)) - 1] = i;
      }
    }
  }
} // namespace dipcc

// Decode a [B, 7, S]-shape tensor of EOS_IDX-padded order idxs.
// Returns a 3d vector of string (batch, power, orders)
vector<vector<vector<string>>>
OrdersDecoder::decode_order_idxs(torch::Tensor *order_idxs) const {
  auto accessor = order_idxs->accessor<long, 3>();
  long batch_size = accessor.size(0);
  long max_seq_len = accessor.size(2);

  vector<vector<vector<string>>> r(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    auto &rb = r[b];
    rb.resize(7);

    for (int p = 0; p < 7; ++p) {
      auto &rbp = rb[p];
      rbp.reserve(max_seq_len);

      for (int i = 0; i < max_seq_len; ++i) {
        long order_idx = accessor[b][p][i];
        if (order_idx == OrdersEncoder::EOS_IDX) {
          continue;
        }
        string order = order_vocabulary_[order_idx];
        for (size_t start = 0, end = 0; end != string::npos; start = end + 1) {
          end = order.find(';', start);
          rbp.push_back(order.substr(start, end - start));
        }
      }
      std::sort(rbp.begin(), rbp.end(), loc_order_cmp);
    }
  }

  return r;
} // decode_order_idxs

vector<vector<vector<string>>> OrdersDecoder::decode_order_idxs_all_powers(
    torch::Tensor *order_idxs, torch::Tensor *x_in_adj_phase,
    torch::Tensor *x_power, int batch_repeat_interleave) const {

  auto accessor_in_adj = x_in_adj_phase->accessor<float, 1>();
  auto accessor_power = x_power->accessor<long, 3>();

  auto order_strings = decode_order_idxs(order_idxs);
  // return order_strings;
  for (size_t output_index = 0; output_index < order_strings.size();
       ++output_index) {
    const long batch_index = output_index / batch_repeat_interleave;
    const bool in_adj = accessor_in_adj[batch_index] > 0.5;
    if (in_adj)
      continue;

    auto &power_orders = order_strings[output_index];
    vector<string> joint_orders;
    joint_orders.swap(power_orders[0]);
    for (int order_index = 0; order_index < joint_orders.size();
         ++order_index) {
      const long power = accessor_power[batch_index][0][order_index];
      if (power == -1)
        break;
      power_orders[power].push_back(joint_orders[order_index]);
    }
  }

  return order_strings;
} // decode_order_idxs_all_powers

vector<int>
OrdersEncoder::filter_orders_in_vocab(const set<Order> &orders) const {
  vector<int> idxs;
  idxs.reserve(orders.size());
  if (allow_buggy_duplicates_) {
    for (const Order &order : orders) {
      int idx = smarter_order_index(order);
      if (idx != -1) {
        idxs.push_back(idx);
      }
    }
  } else {
    // Order-preserving dedup
    std::unordered_set<int> idxs_used;
    idxs_used.reserve(orders.size());
    for (const Order &order : orders) {
      int idx = smarter_order_index(order);
      if (idx != -1 && idxs_used.find(idx) == idxs_used.end()) {
        idxs.push_back(idx);
        idxs_used.insert(idx);
      }
    }
  }
  return idxs;
}

int OrdersEncoder::smarter_order_index(const Order &order) const {
  string order_s(order.to_string());
  auto it = order_vocabulary_to_idx_.find(order_s);
  if (it != order_vocabulary_to_idx_.end()) {
    return it->second;
  }

  // Try order with no coasts
  string order_s_no_coasts;
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

template <typename T>
vector<Loc> OrdersEncoder::get_sorted_actual_orderable_locs(
    const T &root_locs,
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders)
    const {
  vector<Loc> locs;
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

  sort(locs.begin(), locs.end());
  return locs;
}

// See combinations()
void combinations_impl(int min, int n, int c, vector<int> &v,
                       function<void(const vector<int> &)> foo) {
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
void combinations(int n, int c, function<void(const vector<int> &)> foo) {
  JCHECK(n >= c, "Called combinations with n < c");
  vector<int> v;
  v.reserve(c);
  combinations_impl(0, n, c, v, foo);
}

vector<string> get_compound_build_orders(
    const unordered_map<dipcc::Loc, set<dipcc::Order>> &all_possible_orders,
    vector<Loc> orderable_locs, int n_builds) {

  vector<string> r;
  r.reserve(64);

  combinations(orderable_locs.size(), n_builds,
               [&](const vector<int> &orderable_locs_idxs) {
                 vector<vector<Order>> order_lists;
                 order_lists.resize(n_builds);
                 int product = 1;

                 for (int i = 0; i < n_builds; ++i) {

                   for (Loc loc :
                        expand_coasts(orderable_locs[orderable_locs_idxs[i]])) {
                     for (const Order &order : all_possible_orders.at(loc)) {
                       order_lists[i].push_back(order);
                     }
                   }
                   product *= order_lists[i].size();
                 }

                 vector<int> counter(n_builds, 0);

                 vector<string> orders_to_cat;
                 orders_to_cat.resize(n_builds);

                 for (int i = 0; i < product; ++i) {

                   // Gather orders to cat
                   for (int j = 0; j < n_builds; ++j) {
                     orders_to_cat[j] =
                         (order_lists[j][counter[j]].to_string());
                   }
                   sort(orders_to_cat.begin(), orders_to_cat.end());

                   // Cat orders and add to final list
                   string cat(orders_to_cat[0]);
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
