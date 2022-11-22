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

// Version 2
#define SV2_ARMY 0
#define SV2_FLEET 1
#define SV2_AUS 2
#define SV2_ENG 3
#define SV2_FRA 4
#define SV2_ITA 5
#define SV2_GER 6
#define SV2_RUS 7
#define SV2_TUR 8
#define SV2_BUILDABLE 9
#define SV2_REMOVABLE 10
#define SV2_DIS_ARMY 11
#define SV2_DIS_FLEET 12
#define SV2_DIS_AUS 13
#define SV2_DIS_ENG 14
#define SV2_DIS_FRA 15
#define SV2_DIS_ITA 16
#define SV2_DIS_GER 17
#define SV2_DIS_RUS 18
#define SV2_DIS_TUR 19
#define SV2_LAND 20
#define SV2_WATER 21
#define SV2_COAST 22
#define SV2_SC_AUS 23
#define SV2_SC_ENG 24
#define SV2_SC_FRA 25
#define SV2_SC_ITA 26
#define SV2_SC_GER 27
#define SV2_SC_RUS 28
#define SV2_SC_TUR 29
#define SV2_SC_POW_NONE 30
#define SV2_HOME_AUS 31
#define SV2_HOME_ENG 32
#define SV2_HOME_FRA 33
#define SV2_HOME_ITA 34
#define SV2_HOME_GER 35
#define SV2_HOME_RUS 36
#define SV2_HOME_TUR 37

// These macros expect a local variable _enc_width to be present within the
// scope that they are used which should say how many feature channels are
// present at each location. i = location on diplomacy map j = feature channel
#define P_BOARD_STATE(r, i, j) (*((r) + ((i)*_enc_width) + (j)))

namespace py = pybind11;

namespace dipcc {

void encode_board_state_pperm_matrix_v2(const int *pperm, float *r) {
  constexpr int input_version = 2;
  int _enc_width = board_state_enc_width(input_version);
  memset(r, 0, _enc_width * _enc_width * sizeof(float));

  P_BOARD_STATE(r, SV2_ARMY, SV2_ARMY) = 1.0;
  P_BOARD_STATE(r, SV2_FLEET, SV2_FLEET) = 1.0;
  for (int p = 0; p < 7; ++p) {
    P_BOARD_STATE(r, SV2_AUS + p, SV2_AUS + pperm[p]) = 1.0;
  }
  P_BOARD_STATE(r, SV2_BUILDABLE, SV2_BUILDABLE) = 1.0;
  P_BOARD_STATE(r, SV2_REMOVABLE, SV2_REMOVABLE) = 1.0;
  P_BOARD_STATE(r, SV2_DIS_ARMY, SV2_DIS_ARMY) = 1.0;
  P_BOARD_STATE(r, SV2_DIS_FLEET, SV2_DIS_FLEET) = 1.0;
  for (int p = 0; p < 7; ++p) {
    P_BOARD_STATE(r, SV2_DIS_AUS + p, SV2_DIS_AUS + pperm[p]) = 1.0;
  }
  P_BOARD_STATE(r, SV2_LAND, SV2_LAND) = 1.0;
  P_BOARD_STATE(r, SV2_WATER, SV2_WATER) = 1.0;
  P_BOARD_STATE(r, SV2_COAST, SV2_COAST) = 1.0;
  for (int p = 0; p < 7; ++p) {
    P_BOARD_STATE(r, SV2_SC_AUS + p, SV2_SC_AUS + pperm[p]) = 1.0;
  }
  P_BOARD_STATE(r, SV2_SC_POW_NONE, SV2_SC_POW_NONE) = 1.0;
  for (int p = 0; p < 7; ++p) {
    P_BOARD_STATE(r, SV2_HOME_AUS + p, SV2_HOME_AUS + pperm[p]) = 1.0;
  }
}

std::vector<int> encoding_unit_ownership_idxs_v2() {
  return {SV2_AUS, SV2_ENG, SV2_FRA, SV2_ITA, SV2_GER, SV2_RUS, SV2_TUR};
}

std::vector<int> encoding_sc_ownership_idxs_v2() {
  return {SV2_SC_AUS, SV2_SC_ENG, SV2_SC_FRA, SV2_SC_ITA,
          SV2_SC_GER, SV2_SC_RUS, SV2_SC_TUR};
}

void encode_board_state_v2(GameState &state, float *r) {
  constexpr int input_version = 2;
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

    int power_i = static_cast<int>(unit.power) - 1;

    size_t loc_i = static_cast<int>(unit.loc) - 1;
    P_BOARD_STATE(r, loc_i,
                  unit.type == UnitType::ARMY ? SV2_ARMY : SV2_FLEET) = 1;
    P_BOARD_STATE(r, loc_i, SV2_AUS + power_i) = 1;
    P_BOARD_STATE(r, loc_i, SV2_REMOVABLE) = static_cast<float>(removable);
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      P_BOARD_STATE(r, rloc_i,
                    unit.type == UnitType::ARMY ? SV2_ARMY : SV2_FLEET) = 1;
      P_BOARD_STATE(r, rloc_i, SV2_AUS + power_i) = 1;
      P_BOARD_STATE(r, rloc_i, SV2_REMOVABLE) = static_cast<float>(removable);
      filled[rloc_i] = true;
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
        P_BOARD_STATE(r, loc_i, SV2_BUILDABLE) = 1;
      }
    }
  }

  /////////////////////
  // dislodged units //
  /////////////////////

  std::fill(filled.begin(), filled.end(), false);
  for (OwnedUnit unit : state.get_dislodged_units()) {
    size_t loc_i = static_cast<int>(unit.loc) - 1;
    int power_i = static_cast<int>(unit.power) - 1;
    P_BOARD_STATE(r, loc_i,
                  unit.type == UnitType::ARMY ? SV2_DIS_ARMY : SV2_DIS_FLEET) =
        1;
    P_BOARD_STATE(r, loc_i, SV2_DIS_AUS + power_i) = 1;
    filled[loc_i] = true;

    // Mark parent if it's a coast
    Loc rloc = root_loc(unit.loc);
    if (unit.loc != rloc) {
      size_t rloc_i = static_cast<int>(rloc) - 1;
      P_BOARD_STATE(r, rloc_i,
                    unit.type == UnitType::ARMY ? SV2_DIS_ARMY
                                                : SV2_DIS_FLEET) = 1;
      P_BOARD_STATE(r, rloc_i, SV2_DIS_AUS + power_i) = 1;
      filled[rloc_i] = true;
    }
  }

  ///////////////
  // Area type //
  ///////////////

  for (int i = 0; i < 81; ++i) {
    Loc loc = LOCS[i];
    if (is_water(loc)) {
      P_BOARD_STATE(r, i, SV2_WATER) = 1;
    } else if (is_coast(loc)) {
      P_BOARD_STATE(r, i, SV2_COAST) = 1;
    } else {
      P_BOARD_STATE(r, i, SV2_LAND) = 1;
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
    int power_i = power == Power::NONE ? 7 : (static_cast<int>(power) - 1);

    for (Loc cloc : expand_coasts(loc)) {
      int cloc_i = static_cast<int>(cloc) - 1;
      P_BOARD_STATE(r, cloc_i, SV2_SC_AUS + power_i) = 1;
    }
  }

  /////////////////
  // home center //
  /////////////////
  for (int p = 0; p < 7; ++p) {
    Power power = POWERS[p];
    const std::vector<Loc> &hcs = home_centers(power);
    for (Loc loc : hcs) {
      int loc_i = static_cast<int>(loc) - 1;
      int power_i = static_cast<int>(power) - 1;
      P_BOARD_STATE(r, loc_i, SV2_HOME_AUS + power_i) = 1;
    }
  }

} // encode_board_state_v2

} // namespace dipcc
