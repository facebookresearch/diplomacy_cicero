/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <exception>
#include <glog/logging.h>

#include "checks.h"
#include "enums.h"
#include "loc.h"
#include "unit.h"

namespace dipcc {

Unit::Unit(UnitType type, Loc loc) {
  this->type = type;
  this->loc = loc;
}

Unit::Unit(const std::string &s) {
  if (s.at(0) == 'A') {
    this->type = UnitType::ARMY;
  } else if (s.at(0) == 'F') {
    this->type = UnitType::FLEET;
  } else {
    throw("Bad unit: " + s);
  }
  if (s.at(1) != ' ') {
    throw("Bad unit: " + s);
  }
  this->loc = loc_from_str(s.substr(2));
}

std::string Unit::to_string() const {
  JCHECK(this->type != UnitType::NONE,
         "Called NONE Unit to_string, Loc=" + loc_str(this->loc));
  JCHECK(this->loc != Loc::NONE, "Called NONE Unit to_string");

  std::string s;
  s += this->type == UnitType::ARMY ? 'A' : 'F';
  s += ' ';
  s += loc_str(this->loc);
  return s;
}

OwnedUnit Unit::owned_by(Power power) const {
  return OwnedUnit{power, this->type, this->loc};
}

// Comparator (to enable use as set/map key)
std::tuple<UnitType, Loc> Unit::to_tuple() const { return std::tie(type, loc); }
bool Unit::operator<(const Unit &other) const {
  return this->to_tuple() < other.to_tuple();
}

bool Unit::operator==(const Unit &other) const {
  return this->type == other.type && this->loc == other.loc;
}

std::ostream &operator<<(std::ostream &os, const Unit &u) {
  if (u.type == UnitType::NONE) {
    return os << "Unit::NONE";
  } else {
    return os << u.to_string();
  }
}

} // namespace dipcc
