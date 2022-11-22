/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <string>
#include <unordered_map>
#include <vector>

#include "../cc/game.h"

namespace py = pybind11;

using namespace std;

namespace dipcc {

// PUBLIC

py::dict Game::py_get_logs() {
  py::dict d;
  for (auto &[phase, datas] : logs_) {
    py::list l;
    for (const std::shared_ptr<const std::string> &data : datas) {
      l.append(*data);
    }
    d[py::cast<std::string>(phase.to_string())] = l;
  }
  return d;
}

} // namespace dipcc
