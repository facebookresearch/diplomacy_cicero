/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "enums.h"
#include "hash.h"
#include "order.h"
#include "owned_unit.h"
#include "phase.h"
#include "power.h"
#include "scoring.h"
#include "thirdparty/nlohmann/json.hpp"

namespace dipcc {

struct Resolution; // defined in process.cc

class GameState {
public:
  GameState(){};
  GameState(const json &j);

  OwnedUnit get_unit(Loc loc) const;
  OwnedUnit get_unit_rooted(Loc loc) const;
  void set_unit(Power power, UnitType type, Loc loc);
  void set_units(const std::map<Loc, OwnedUnit> units) {
    units_ = units;
    influence_.clear();
    for (const auto &[loc, unit] : units_)
      influence_[loc] = unit.power;
  }
  void set_influence(const std::map<Loc, Power> &influence) {
    influence_ = influence;
  }
  void remove_unit_rooted(Loc);
  const std::map<Loc, OwnedUnit> &get_units() const { return units_; }
  const std::map<Loc, Power> &get_influence() const { return influence_; }

  void set_center(Loc loc, Power power);
  void set_centers(const std::map<Loc, Power> &centers) { centers_ = centers; }
  const std::map<Loc, Power> &get_centers() const { return centers_; }

  Phase get_phase() const { return phase_; }
  void set_phase(Phase phase) { phase_ = phase; }

  void add_dislodged_unit(OwnedUnit unit, Loc dislodged_by);
  void remove_dislodged_unit(OwnedUnit unit);
  void add_contested_loc(Loc loc);
  std::vector<OwnedUnit> get_dislodged_units() const;
  int get_n_builds(Power power);

  const std::unordered_map<Power, std::vector<Loc>> &get_orderable_locations();
  const std::unordered_map<Loc, std::set<Order>> &get_all_possible_orders();
  void clear_all_possible_orders();

  void do_civil_disorder(Power power, int n);
  void maybe_skip_winter_or_finish();

  std::vector<float> get_scores(Scoring scoring_system) const;

  bool any_sc_occupied_by_new_power() const;

  GameState process(const std::unordered_map<Power, std::vector<Order>> &orders,
                    bool exception_on_convoy_paradox = false);

  nlohmann::json to_json();

  size_t compute_board_hash() const;

private:
  void load_all_possible_orders_m();
  void load_all_possible_orders_r();
  void load_all_possible_orders_a();
  void copy_possible_orders_to_root_loc();
  void
  copy_sorted_root_locs(const std::unordered_map<Power, std::set<Loc>> &from,
                        std::unordered_map<Power, std::vector<Loc>> &to);

  GameState
  process_m(const std::unordered_map<Power, std::vector<Order>> &orders,
            bool exception_on_convoy_paradox = false);
  GameState
  process_r(const std::unordered_map<Power, std::vector<Order>> &orders);
  GameState
  process_a(const std::unordered_map<Power, std::vector<Order>> &orders);

  bool has_any_unoccupied_home(Power power) const;
  void recalculate_centers();
  Power get_winner() const;
  GameState build_next_state(const Resolution &) const;
  friend void to_json(json &j, const GameState &x);

  void debug_log_all_possible_orders();

  // Members
  Phase phase_ = {'S', 1901, 'M'};
  std::map<Loc, OwnedUnit> units_;
  std::map<Loc, Power> centers_;
  std::map<Loc, Power> influence_;           // only for vizualization purposes.
  std::map<OwnedUnit, Loc> dislodged_units_; // only valid during R phase
  std::set<Loc> contested_locs_;             // only valid during R phase
  std::vector<int> n_builds_; // 7-len vector only valid during A phase

  std::unordered_map<Loc, std::set<Order>> all_possible_orders_;
  std::unordered_map<Power, std::vector<Loc>> orderable_locations_;
  bool orders_loaded_ = false;

  const static int MAX_YEAR = 1935;
};

} // namespace dipcc
