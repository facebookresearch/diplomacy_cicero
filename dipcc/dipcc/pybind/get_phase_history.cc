/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <string>
#include <vector>

#include "../cc/game.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

vector<PhaseData> Game::get_phase_history() {
  vector<PhaseData> r;
  r.reserve(state_history_.size());

  for (auto &it : state_history_) {
    string name = it.first.to_string();
    r.push_back(
        PhaseData(*it.second, *order_history_[name], message_history_[name]));
  }

  return r;
}

vector<PhaseData> Game::get_all_phases() {
  vector<PhaseData> r = get_phase_history();
  std::map<uint64_t, Message> phase_messages;
  if (message_history_.find(state_->get_phase()) != message_history_.end()) {
    phase_messages = message_history_[state_->get_phase()];
  }
  r.push_back(PhaseData(*state_, staged_orders_, phase_messages));

  return r;
}

vector<std::string> Game::get_all_phase_names() const {
  vector<std::string> r;
  r.reserve(state_history_.size() + 1);
  for (auto &it : state_history_) {
    string name = it.first.to_string();
    r.push_back(name);
  }
  r.push_back(state_->get_phase().to_string());
  return r;
}

} // namespace dipcc
