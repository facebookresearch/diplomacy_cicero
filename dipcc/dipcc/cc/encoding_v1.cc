/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "encoding.h"

#include <glog/logging.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "checks.h"
#include "game.h"
#include "game_state.h"
#include "loc.h"
#include "power.h"

// Version 1
#define S_ARMY 0
#define S_FLEET 1
#define S_UNIT_NONE 2
#define S_AUS 3
#define S_ENG 4
#define S_FRA 5
#define S_ITA 6
#define S_GER 7
#define S_RUS 8
#define S_TUR 9
#define S_POW_NONE 10
#define S_BUILDABLE 11
#define S_REMOVABLE 12
#define S_DIS_ARMY 13
#define S_DIS_FLEET 14
#define S_DIS_UNIT_NONE 15
#define S_DIS_AUS 16
#define S_DIS_ENG 17
#define S_DIS_FRA 18
#define S_DIS_ITA 19
#define S_DIS_GER 20
#define S_DIS_RUS 21
#define S_DIS_TUR 22
#define S_DIS_POW_NONE 23
#define S_LAND 24
#define S_WATER 25
#define S_COAST 26
#define S_SC_AUS 27
#define S_SC_ENG 28
#define S_SC_FRA 29
#define S_SC_ITA 30
#define S_SC_GER 31
#define S_SC_RUS 32
#define S_SC_TUR 33
#define S_SC_POW_NONE 34

// These macros expect a local variable _enc_width to be present within the
// scope that they are used which should say how many feature channels are
// present at each location. i = location on diplomacy map j = feature channel
#define P_BOARD_STATE(r, i, j) (*((r) + ((i)*_enc_width) + (j)))

namespace py = pybind11;

namespace dipcc {

std::vector<int> encoding_unit_ownership_idxs_v1() {
  return {S_AUS, S_ENG, S_FRA, S_ITA, S_GER, S_RUS, S_TUR};
}

std::vector<int> encoding_sc_ownership_idxs_v1() {
  return {S_SC_AUS, S_SC_ENG, S_SC_FRA, S_SC_ITA, S_SC_GER, S_SC_RUS, S_SC_TUR};
}

void encode_board_state_v1(GameState &state, float *r) {
  constexpr int input_version = 1;
  int _enc_width = board_state_enc_width(input_version);
  memset(r, 0, 81 * _enc_width * sizeof(float));

  //////////////////////////////////////
  // unit type, unit power, removable //
  //////////////////////////////////////

  std::vector<bool> filled(81, false);

  for (auto p : state.get_units()) {
    OwnedUnit unit = p.second;
    JCHECK(unit.type != UnitType::NONE, "UnitType::NONE");
    JCHECK(unit.loc != Loc::NONE, "Loc::NONE");
    JCHECK(unit.power != Power::NONE, "Power::NONE");

    bool removable =
        state.get_phase().season == 'W' && state.get_n_builds(unit.power) < 0;

    size_t loc_i = static_cast<int>(unit.loc) - 1;
    P_BOARD_STATE(r, loc_i, unit.type == UnitType::ARMY ? S_ARMY : S_FLEET) = 1;
    P_BOARD_STATE(r, loc_i, S_AUS + static_cast<int>(unit.power) - 1) = 1;
    P_BOARD_STATE(r, loc_i, S_REMOVABLE) = static_cast<float>(removable);
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      P_BOARD_STATE(r, rloc_i, unit.type == UnitType::ARMY ? S_ARMY : S_FLEET) =
          1;
      P_BOARD_STATE(r, rloc_i, S_AUS + static_cast<int>(unit.power) - 1) = 1;
      P_BOARD_STATE(r, rloc_i, S_REMOVABLE) = static_cast<float>(removable);
      filled[rloc_i] = true;
    }
  }

  // Set locs with no units
  for (int i = 0; i < 81; ++i) {
    if (!filled[i]) {
      P_BOARD_STATE(r, i, S_UNIT_NONE) = 1;
      P_BOARD_STATE(r, i, S_POW_NONE) = 1;
    }
  }

  ///////////////
  // buildable //
  ///////////////

  if (state.get_phase().phase_type == 'A') {
    for (auto &p : state.get_all_possible_orders()) {
      auto order = p.second.begin();
      if (order->get_type() == OrderType::B) {
        Loc loc = order->get_unit().loc;
        size_t loc_i = static_cast<int>(loc) - 1;
        P_BOARD_STATE(r, loc_i, S_BUILDABLE) = 1;
      }
    }
  }

  /////////////////////
  // dislodged units //
  /////////////////////

  std::fill(filled.begin(), filled.end(), false);
  for (OwnedUnit unit : state.get_dislodged_units()) {
    size_t loc_i = static_cast<int>(unit.loc) - 1;
    P_BOARD_STATE(r, loc_i,
                  unit.type == UnitType::ARMY ? S_DIS_ARMY : S_DIS_FLEET) = 1;
    P_BOARD_STATE(r, loc_i, S_DIS_AUS + static_cast<int>(unit.power) - 1) = 1;
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      P_BOARD_STATE(r, rloc_i,
                    unit.type == UnitType::ARMY ? S_DIS_ARMY : S_DIS_FLEET) = 1;
      P_BOARD_STATE(r, rloc_i, S_DIS_AUS + static_cast<int>(unit.power) - 1) =
          1;
      filled[rloc_i] = true;
    }
  }

  // Set locs with no dislodged units
  for (int i = 0; i < 81; ++i) {
    if (!filled[i]) {
      P_BOARD_STATE(r, i, S_DIS_UNIT_NONE) = 1;
      P_BOARD_STATE(r, i, S_DIS_POW_NONE) = 1;
    }
  }

  ///////////////
  // Area type //
  ///////////////

  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (is_water(loc)) {
      P_BOARD_STATE(r, i, S_WATER) = 1;
    } else if (is_coast(loc)) {
      P_BOARD_STATE(r, i, S_COAST) = 1;
    } else {
      P_BOARD_STATE(r, i, S_LAND) = 1;
    }
  }

  ///////////////////
  // supply center //
  ///////////////////

  auto centers = state.get_centers();
  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (!is_center(loc) || loc != root_loc(loc)) {
      continue;
    }

    auto it = centers.find(loc);
    Power power = it == centers.end() ? Power::NONE : it->second;
    int off = power == Power::NONE ? 7 : static_cast<int>(power) - 1;

    for (Loc cloc : expand_coasts(loc)) {
      int cloc_i = static_cast<int>(cloc) - 1;
      P_BOARD_STATE(r, cloc_i, S_SC_AUS + off) = 1;
    }
  }

} // encode_board_state_v1

} // namespace dipcc
