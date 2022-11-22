/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <glog/logging.h>

#include "checks.h"
#include "power.h"

namespace dipcc {

std::string power_str(const Power &power) {
  JCHECK(power != Power::NONE, "power_str got None");
  return POWERS_STR.at(static_cast<size_t>(power) - 1); // -1 for NONE
}

std::string power_or_all_str(const PowerOrAll &power_or_all) {
  JCHECK(power_or_all != PowerOrAll::NONE, "power_or_all_str got None");
  return POWERS_OR_ALL_STR.at(static_cast<size_t>(power_or_all) -
                              1); // -1 for NONE
}

Power power_from_str(const std::string &s) {
  for (int i = 0; i < NUM_POWERS; ++i) {
    if (power_str(POWERS[i]) == s) {
      return POWERS[i];
    }
  }
  JFAIL("Bad arg to power_from_str: " + s);
}

PowerOrAll power_or_all_from_str(const std::string &s) {
  for (int i = 0; i < NUM_POWERS_OR_ALL; ++i) {
    if (power_or_all_str(POWERS_OR_ALL[i]) == s) {
      return POWERS_OR_ALL[i];
    }
  }
  JFAIL("Bad arg to power_or_all_from_str: " + s);
}

} // namespace dipcc
