/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>

#include "enums.h"
#include "loc.h"
#include "owned_unit.h"
#include "power.h"

namespace dipcc {

struct OwnedUnit; // forward declare

struct Unit {
  UnitType type = UnitType::NONE;
  Loc loc = Loc::NONE;

  Unit() {}
  Unit(UnitType type, Loc loc);
  Unit(const std::string &s);

  std::string to_string() const;

  // Conversion to OwnedUnit
  OwnedUnit owned_by(Power power) const;

  // Comparator (to enable use as set/map key)
  std::tuple<UnitType, Loc> to_tuple() const;
  bool operator<(const Unit &other) const;

  // Equality operator, true if members equal
  bool operator==(const Unit &other) const;

  // Print operator
  friend std::ostream &operator<<(std::ostream &, const Unit &);
};
void to_json(json &j, const Unit &x);

} // namespace dipcc
