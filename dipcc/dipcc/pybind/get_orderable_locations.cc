/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <string>
#include <vector>

#include "../cc/game.h"
#include "../cc/loc.h"
#include "phase_data.h"

using namespace std;

namespace dipcc {

py::dict Game::py_get_orderable_locations() {
  py::dict d;

  for (Power power : POWERS) {
    d[py::cast<string>(power_str(power))] = py::list();
  }

  for (auto &[power, locs] : this->get_orderable_locations()) {
    auto power_s = py::cast<string>(power_str(power));
    py::list list;
    for (Loc loc : locs) {
      list.append(loc_str(loc));
    }
    d[power_s] = list;
  }

  return d;
}

} // namespace dipcc
