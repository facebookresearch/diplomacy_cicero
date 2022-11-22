/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "enums.h"
#include "checks.h"

namespace dipcc {

std::ostream &operator<<(std::ostream &os, UnitType t) {
  switch (t) {
  case UnitType::NONE:
    return os << "NONE";
  case UnitType::ARMY:
    return os << "ARMY";
  case UnitType::FLEET:
    return os << "FLEET";
  default:
    JFAIL("unknown unit type");
  }
}

} // namespace dipcc
