/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "../pybind/phase_data.h"
#include "../pybind/py_dict.h"
#include "enums.h"
#include "game_state.h"
#include "hash.h"
#include "json.h"
#include "loc.h"
#include "message.h"
#include "order.h"
#include "phase.h"
#include "power.h"
#include "scoring.h"
#include "thirdparty/nlohmann/json.hpp"
#include "unit.h"

using nlohmann::json;

namespace dipcc {

class Game {
public:
  Game(int draw_on_stalemate_years = -1, bool is_full_press = true);
  Game(const std::string &json_str);

  void set_orders(const std::string &power,
                  const std::vector<std::string> &orders);

  void set_all_orders(
      const std::map<std::string, std::vector<std::string>> &orders_by_power);

  void clear_orders() { staged_orders_.clear(); };

  void process();

  GameState &get_state();
  const GameState &get_state() const;

  std::unordered_map<Power, std::vector<Loc>> get_orderable_locations();

  const std::unordered_map<Loc, std::set<Order>> &get_all_possible_orders();

  bool is_game_done() const;

  GameState *get_last_movement_phase(); // can return nullptr

  std::optional<Phase> get_next_phase(Phase from);
  std::optional<Phase> get_prev_phase(Phase from);

  std::string game_id;

  std::string to_json();

  Game rolled_back_to_phase_start(const std::string &phase_s);
  Game rolled_back_to_phase_end(const std::string &phase_s);

  // Rolls back to phase of the last message <= timestamp, not preserving the
  // staged orders on that phase. rolled_back_to_timestamp_start keeps messages
  // < timestamp rolled_back_to_timestamp_end keeps messages <= timestamp
  Game rolled_back_to_timestamp_start(const uint64_t timestamp);
  Game rolled_back_to_timestamp_end(const uint64_t timestamp);

  // Returns the first phase in the game if there are no messages at or before
  // timestamp.
  Phase phase_of_last_message_at_or_before(const uint64_t timestamp) const;
  std::string
  py_phase_of_last_message_at_or_before(const uint64_t timestamp) const;

  void rollback_messages_to_timestamp_start(const uint64_t timestamp);
  void rollback_messages_to_timestamp_end(const uint64_t timestamp);

  void delete_message_at_timestamp(const uint64_t timestamp);

  uint64_t get_last_message_timestamp() const;

  std::map<Phase, std::shared_ptr<GameState>> &get_state_history() {
    return state_history_;
  }
  std::map<Phase,
           std::shared_ptr<const std::unordered_map<Power, std::vector<Order>>>>
      &get_order_history() {
    return order_history_;
  }
  const std::map<Phase, std::shared_ptr<GameState>> &get_state_history() const {
    return state_history_;
  }
  const std::map<
      Phase,
      std::shared_ptr<const std::unordered_map<Power, std::vector<Order>>>> &
  get_order_history() const {
    return order_history_;
  }

  size_t compute_board_hash() const { return state_->compute_board_hash(); }
  size_t compute_order_history_hash() const;

  std::vector<float> get_scores() const {
    return state_->get_scores(scoring_system_);
  }
  std::vector<float> get_scores(Scoring override_scoring_system) const {
    return state_->get_scores(override_scoring_system);
  }

  void clear_old_all_possible_orders();

  void set_exception_on_convoy_paradox() {
    exception_on_convoy_paradox_ = true;
  }

  int get_consecutive_years_without_sc_change() const;
  bool any_sc_occupied_by_new_power() const {
    return state_->any_sc_occupied_by_new_power();
  }
  void set_draw_on_stalemate_years(int year) {
    draw_on_stalemate_years_ = year;
  }

  // press

  bool is_full_press() const { return is_full_press_; }

  std::map<Phase, std::map<uint64_t, Message>> &get_message_history() {
    return message_history_;
  }

  void add_message(Power sender, PowerOrAll recipient, const std::string &body,
                   uint64_t time_sent, bool increment_on_collision = false);

  // metadata
  void set_metadata(const std::string &k, const std::string &v) {
    metadata_[k] = v;
  }

  std::string &get_metadata(const std::string &k) { return metadata_[k]; }

  // python

  std::unordered_map<std::string, std::vector<std::string>>
  py_get_all_possible_orders();
  pybind11::dict py_get_state();
  pybind11::dict py_get_orderable_locations();
  std::vector<PhaseData> get_phase_history();
  PhaseData get_phase_data()
      const; // Deliberately weird - does NOT return staged orders and messages
  PhaseData get_staged_phase_data()
      const; // Non-weird, does return staged orders and messages

  // Same as get_phase_history but also includes the current phase, and does NOT
  // share the same weird hack of get_phase_data where staged orders and
  // messages are omitted.
  std::vector<PhaseData> get_all_phases();
  std::vector<std::string> get_all_phase_names() const;

  pybind11::dict py_get_message_history();
  pybind11::dict py_get_messages();
  pybind11::dict py_get_orders() { return py_orders_to_dict(staged_orders_); }

  pybind11::dict py_get_logs();
  void add_log(const std::string &body);

  Scoring get_scoring_system() const;
  void set_scoring_system(Scoring scoring);

  static Game from_json(const std::string &s) { return Game(s); }

  std::string get_phase_long() { return state_->get_phase().to_string_long(); }
  std::string get_phase_short() { return state_->get_phase().to_string(); }
  uint32_t get_year() { return state_->get_phase().year; }
  void py_add_message(const std::string &sender, const std::string &recipient,
                      const std::string &body, uint64_t time_sent,
                      bool increment_on_collision = false) {
    add_message(power_from_str(sender), power_or_all_from_str(recipient), body,
                time_sent, increment_on_collision);
  }

  // Some convenience functions for accessing individual location data on the
  // map

  // Returns the owning power and/or unit type if it exists, returns None if no
  // unit there. If a location is specified without coasts, then tests that
  // location and all possible coastal extensions of it. If a location is
  // specified with a coast, only finds the unit if it is there at that exact
  // coast.
  std::optional<std::string> get_unit_power_at(const std::string &loc_str);
  std::optional<std::string> get_unit_type_at(const std::string &loc_str);

  bool is_supply_center(const std::string &loc_str);
  // Returns the owning power of the given SC, None if not an SC or if no power
  // owns it. Coastal version of SCs are accepted as well and treated the same
  // as the root location.
  std::optional<std::string>
  get_supply_center_power(const std::string &loc_str);

  // mila compat

  std::string map_name() { return map_name_; }
  char phase_type() { return state_->get_phase().phase_type; }

private:
  void crash_dump();
  void maybe_early_exit();

  void rollback_to_phase(Phase phase, bool preserve_phase_messages,
                         bool preserve_phase_orders, bool preserve_phase_logs);

  // Members
  std::shared_ptr<GameState> state_;
  std::unordered_map<Power, std::vector<Order>> staged_orders_;
  std::map<Phase, std::shared_ptr<GameState>> state_history_;
  std::map<Phase,
           std::shared_ptr<const std::unordered_map<Power, std::vector<Order>>>>
      order_history_;
  std::map<Phase, std::vector<std::shared_ptr<const std::string>>> logs_;
  std::map<Phase, std::map<uint64_t, Message>> message_history_;
  int draw_on_stalemate_years_ = -1;
  bool exception_on_convoy_paradox_ = false;
  std::unordered_map<std::string, std::string> metadata_;
  Scoring scoring_system_ = Scoring::SOS;
  bool is_full_press_ = true;
  std::string map_name_ = "standard";
};

} // namespace dipcc
