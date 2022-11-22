/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/game.h"
#include "messaging.h"

using namespace std;

namespace dipcc {

// PUBLIC

py::dict Game::py_get_message_history() {
  return py_message_history_to_dict(message_history_, state_->get_phase());
}

py::dict Game::py_get_messages() {
  return py_messages_to_phase_dict(message_history_[state_->get_phase()]);
}

// PRIVATE

py::dict py_message_to_dict(const Message &message) {
  py::object timestamp_init = py::module::import("fairdiplomacy.timestamp")
                                  .attr("Timestamp")
                                  .attr("from_centis");

  py::dict d;
  d["sender"] = power_str(message.sender);
  d["recipient"] = power_or_all_str(message.recipient);
  d["phase"] = message.phase.to_string();
  d["message"] = message.message;
  d["time_sent"] = timestamp_init(message.time_sent);
  return d;
}

py::dict
py_messages_to_phase_dict(const std::map<uint64_t, Message> &messages) {
  py::object timestamp_init = py::module::import("fairdiplomacy.timestamp")
                                  .attr("Timestamp")
                                  .attr("from_centis");

  py::dict d;
  for (const auto &[time_sent, msg] : messages) {
    d[timestamp_init(msg.time_sent)] = py_message_to_dict(msg);
  }
  return d;
}

py::dict py_message_history_to_dict(
    const std::map<Phase, std::map<uint64_t, Message>> &message_history,
    Phase exclude_phase) {

  py::dict d;
  for (auto &[phase, messages] : message_history) {
    if (phase == exclude_phase) {
      continue;
    }
    d[py::cast<std::string>(phase.to_string())] =
        py_messages_to_phase_dict(messages);
  }
  return d;
}

} // namespace dipcc
