/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <glog/logging.h>

#include "checks.h"
#include "enums.h"
#include "loc.h"
#include "owned_unit.h"

namespace dipcc {

Unit OwnedUnit::unowned() const { return Unit(type, loc); }

std::string OwnedUnit::to_string() const {
  JCHECK(this->type != UnitType::NONE, "Called NONE OwnedUnit to_string");
  JCHECK(this->loc != Loc::NONE, "Called NONE OwnedUnit to_string");

  std::string s;
  s += this->type == UnitType::ARMY ? 'A' : 'F';
  s += ' ';
  s += loc_str(this->loc);
  return s;
}

// Comparator (to enable use as set/map key)
std::tuple<Power, UnitType, Loc> OwnedUnit::to_tuple() const {
  return std::tie(power, type, loc);
}
bool OwnedUnit::operator<(const OwnedUnit &other) const {
  return this->to_tuple() < other.to_tuple();
}

bool OwnedUnit::operator==(const OwnedUnit &other) const {
  return this->type == other.type && this->loc == other.loc &&
         this->power == other.power;
}

std::ostream &operator<<(std::ostream &os, const OwnedUnit &u) {
  if (u.type == UnitType::NONE) {
    return os << "OwnedUnit::NONE";
  } else {
    return os << u.to_string() << " (" << power_str(u.power).substr(0, 1)
              << ")";
  }
}

} // namespace dipcc
