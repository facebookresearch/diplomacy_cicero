/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "../cc/checks.h"
#include "../cc/encoding.h"
#include "../cc/thirdparty/nlohmann/json.hpp"

namespace py = pybind11;

namespace dipcc {

py::array_t<float> py_encode_board_state(GameState &state, int input_version) {
  int bwidth = board_state_enc_width(input_version);
  py::array_t<float> r({NUM_LOCS, bwidth});
  encode_board_state(state, input_version, r.mutable_data(0, 0));
  return r;
}

py::array_t<float> py_encode_board_state_from_json(const std::string &json_str,
                                                   int input_version) {
  auto j = json::parse(json_str);
  GameState state(j);
  return py_encode_board_state(state, input_version);
}

py::array_t<float> py_encode_board_state_from_phase(PhaseData &phase,
                                                    int input_version) {
  return py_encode_board_state(phase.get_state(), input_version);
}

py::array_t<float>
py_encode_board_state_pperm_matrices(const py::array_t<int> &pperms,
                                     int input_version) {
  JCHECK(pperms.ndim() == 2,
         "py_encode_board_state_pperm_matrices ndim must be 2");
  JCHECK(pperms.shape(1) == 7,
         "py_encode_board_state_pperm_matrices shape(1) must be 7");
  size_t batch_size = pperms.shape(0);
  int bwidth = board_state_enc_width(input_version);
  py::array_t<float> r({static_cast<py::ssize_t>(pperms.shape(0)),
                        static_cast<py::ssize_t>(bwidth),
                        static_cast<py::ssize_t>(bwidth)});
  for (size_t i = 0; i < batch_size; ++i) {
    encode_board_state_pperm_matrix(pperms.data(i, 0), input_version,
                                    r.mutable_data(i, 0, 0));
  }
  return r;
}

} // namespace dipcc
