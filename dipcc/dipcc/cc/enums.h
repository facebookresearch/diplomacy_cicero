/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>
#include <vector>

#include "thirdparty/nlohmann/json.hpp"

using nlohmann::json;

namespace dipcc {

enum class UnitType {
  NONE,
  ARMY,
  FLEET,
};
NLOHMANN_JSON_SERIALIZE_ENUM(UnitType,
                             {{UnitType::ARMY, "A"}, {UnitType::FLEET, "F"}})
std::ostream &operator<<(std::ostream &os, UnitType t);

enum class OrderType {
  NONE,
  H,
  M,
  SH,
  SM,
  C,
  R,
  B,
  D,
};

} // namespace dipcc
