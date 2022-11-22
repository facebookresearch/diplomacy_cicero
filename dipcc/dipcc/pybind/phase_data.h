/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <unordered_map>

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/message.h"
#include "../cc/power.h"
#include "py_dict.h"

namespace py = pybind11;

namespace dipcc {

// forward declares
py::dict
py_orders_to_dict(std::unordered_map<Power, std::vector<Order>> &orders);
py::dict py_state_to_dict(GameState &state);
py::dict py_messages_to_phase_dict(const std::map<uint64_t, Message> &messages);
// !forward declares

class PhaseData {
public:
  PhaseData(const GameState &state,
            const std::unordered_map<Power, std::vector<Order>> &orders) {
    name_ = state.get_phase().to_string();
    state_ = state;
    orders_ = orders;
  }

  PhaseData(const GameState &state,
            const std::unordered_map<Power, std::vector<Order>> &orders,
            const std::map<uint64_t, Message> &messages) {
    name_ = state.get_phase().to_string();
    state_ = state;
    orders_ = orders;
    messages_ = messages;
  }

  py::dict py_get_state() { return py_state_to_dict(state_); }

  py::dict py_get_orders() { return py_orders_to_dict(orders_); }

  py::dict py_get_messages() { return py_messages_to_phase_dict(messages_); }

  py::dict to_dict() {
    py::dict d;
    d["name"] = name_;
    d["state"] = py_get_state();
    d["orders"] = py_get_orders();
    d["messages"] = py_get_messages();
    return d;
  }

  const std::string get_name() const { return name_; }
  GameState &get_state() { return state_; }
  const std::unordered_map<Power, std::vector<Order>> &get_orders() const {
    return orders_;
  }

  std::vector<float> get_scores(Scoring scoring_system) const {
    return state_.get_scores(scoring_system);
  }

private:
  // Members
  std::string name_;
  GameState state_;
  std::unordered_map<Power, std::vector<Order>> orders_;
  std::map<uint64_t, Message> messages_;
};

} // namespace dipcc
