/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>

#include "enums.h"
#include "loc.h"
#include "power.h"
#include "unit.h"

namespace dipcc {

struct Unit; // forward declare

struct OwnedUnit {
  Power power;
  UnitType type;
  Loc loc;

  std::string to_string() const;

  // Comparator (to enable use as set/map key)
  std::tuple<Power, UnitType, Loc> to_tuple() const;
  bool operator<(const OwnedUnit &other) const;

  // Equality operator, true if members equal
  bool operator==(const OwnedUnit &other) const;
  bool operator!=(const OwnedUnit &other) const { return !operator==(other); }

  // Conversion to unowned unit
  Unit unowned() const;

  // Print operator
  friend std::ostream &operator<<(std::ostream &, const OwnedUnit &);
};
void to_json(json &j, const OwnedUnit &x);

} // namespace dipcc
