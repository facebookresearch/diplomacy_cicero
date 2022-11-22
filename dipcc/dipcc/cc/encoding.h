/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "../cc/game.h"
#include "../cc/game_state.h"
#include "../cc/power.h"

#define PREV_ORDERS_CAPACITY 100

namespace py = pybind11;

namespace dipcc {

// Valid input versions
// Version 1
// The basic input features used for Mila and BaseStrategyModel through mid-2021
//
// Version 2
// Adds a one-hot indicator for each power's home centers in board state
// Cleans up and removes several redundancies in the input encoding from version
// 1, which may slightly speed up encoding and remove a few channels.
//
// Version 3
// Fixes a bug that allowed duplicate coastal orders in order encoder.
// Still uses the same "v2" feature encoding as version 2 for board state.
constexpr int MAX_INPUT_VERSION = 3;

inline constexpr int board_state_enc_width(int input_version) {
  static_assert(MAX_INPUT_VERSION <= 3,
                "Don't forget to update code here if necessary when changing "
                "MAX_INPUT_VERSION");
  assert(input_version >= 1 && input_version <= MAX_INPUT_VERSION);
  return input_version >= 2 ? 38 : 35;
}

void encode_board_state(GameState &state, int input_version, float *r);

// This function, given a permutation pperm of the numbers 0-6, computes a
// permutation matrix M, such that multiplying the board state encoding features
// by M applies that permutation to the 7 powers. For example, if pperm[FRANCE]
// = GERMANY, then torch.matmul(board state encoding, M) will result in a new
// encoding where every unit, SC, etc. that France owns, the encoding will say
// Germany owns.
void encode_board_state_pperm_matrix(const int *pperm, int input_version,
                                     float *r);

// Return the list of feature encoding indices c_i such that if the encoding
// tensor is one-hot at [batch_elt, location, c_i], then power i has a unit at
// location.
std::vector<int> encoding_unit_ownership_idxs(int input_version);
// Same, but for ownership of supply center at location by power i.
std::vector<int> encoding_sc_ownership_idxs(int input_version);

// Implementations for specific versions
void encode_board_state_v1(GameState &state, float *r);
void encode_board_state_v2(GameState &state, float *r);
void encode_board_state_pperm_matrix_v2(const int *pperm, float *r);

std::vector<int> encoding_unit_ownership_idxs_v1();
std::vector<int> encoding_unit_ownership_idxs_v2();

std::vector<int> encoding_sc_ownership_idxs_v1();
std::vector<int> encoding_sc_ownership_idxs_v2();

} // namespace dipcc
