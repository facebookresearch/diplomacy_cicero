/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "encoding.h"

#include "checks.h"
#include "game_state.h"

namespace dipcc {

void encode_board_state(GameState &state, int input_version, float *r) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  if (input_version == 1) {
    encode_board_state_v1(state, r);
  } else if (input_version == 2 || input_version == 3) {
    encode_board_state_v2(state, r);
  } else {
    JFAIL("Unknown input version in encode_board_state");
  }
}

void encode_board_state_pperm_matrix(const int *pperm, int input_version,
                                     float *r) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  if (input_version == 1) {
    JFAIL("PowerPermutation not supported for input_version 1");
  } else if (input_version == 2 || input_version == 3) {
    encode_board_state_pperm_matrix_v2(pperm, r);
  } else {
    JFAIL("Unknown input version in encode_board_state_pperm_matrix");
  }
}

std::vector<int> encoding_unit_ownership_idxs(int input_version) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  if (input_version == 1) {
    return encoding_unit_ownership_idxs_v1();
  } else if (input_version == 2 || input_version == 3) {
    return encoding_unit_ownership_idxs_v2();
  } else {
    JFAIL("Unknown input version in encoding_unit_ownership_idxs");
  }
}

std::vector<int> encoding_sc_ownership_idxs(int input_version) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  if (input_version == 1) {
    return encoding_sc_ownership_idxs_v1();
  } else if (input_version == 2 || input_version == 3) {
    return encoding_sc_ownership_idxs_v2();
  } else {
    JFAIL("Unknown input version in encoding_sc_ownership_idxs");
  }
}

} // namespace dipcc
