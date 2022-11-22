/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "enums.h"
#include "game.h"
#include "loc.h"
#include "owned_unit.h"
#include "unit.h"
#include <string>

#include "thirdparty/nlohmann/json.hpp"

using namespace std;
using nlohmann::json;

namespace dipcc {

void to_json(json &j, const Unit &x) { j = x.to_string(); }

void to_json(json &j, const OwnedUnit &x) { j = x.to_string(); }

void to_json(json &j, const Phase &x) { j = x.to_string(); }

void to_json(json &j, const GameState &x) {
  j["phase"] = x.phase_;

  for (auto &it : x.units_) {
    auto &unit = it.second;
    j["units"][power_str(unit.power)].push_back(unit);
  }

  for (auto &it : x.centers_) {
    j["centers"][power_str(it.second)].push_back(loc_str(it.first));
  }
}

void to_json(json &j, const Message &x) {
  j["sender"] = power_str(x.sender);
  j["recipient"] = power_or_all_str(x.recipient);
  j["phase"] = x.phase.to_string();
  j["message"] = x.message;
  j["time_sent"] = x.time_sent;
}

void from_json(const json &j, Message &x) {
  x.sender = power_from_str(j["sender"]);
  x.recipient = power_or_all_from_str(j["recipient"]);
  x.phase = Phase(j["phase"].get<std::string>());
  x.message = j["message"].get<std::string>();
  x.time_sent = j["time_sent"];
}

} // namespace dipcc
