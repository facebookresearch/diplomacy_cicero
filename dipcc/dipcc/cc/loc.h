/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "power.h"
#include "thirdparty/nlohmann/json.hpp"

namespace dipcc {

constexpr int NUM_LOCS = 81;

enum class Loc {
  NONE,
  YOR,
  EDI,
  LON,
  LVP,
  NTH,
  WAL,
  CLY,
  NWG,
  ENG,
  IRI,
  NAO,
  BEL,
  DEN,
  HEL,
  HOL,
  NWY,
  SKA,
  BAR,
  BRE,
  MAO,
  PIC,
  BUR,
  RUH,
  BAL,
  KIE,
  SWE,
  FIN,
  STP,
  STP_NC,
  GAS,
  PAR,
  NAF,
  POR,
  SPA,
  SPA_NC,
  SPA_SC,
  WES,
  MAR,
  MUN,
  BER,
  BOT,
  LVN,
  PRU,
  STP_SC,
  MOS,
  TUN,
  LYO,
  TYS,
  PIE,
  BOH,
  SIL,
  TYR,
  WAR,
  SEV,
  UKR,
  ION,
  TUS,
  NAP,
  ROM,
  VEN,
  GAL,
  VIE,
  TRI,
  ARM,
  BLA,
  RUM,
  ADR,
  AEG,
  ALB,
  APU,
  EAS,
  GRE,
  BUD,
  SER,
  ANK,
  SMY,
  SYR,
  BUL,
  BUL_EC,
  CON,
  BUL_SC
};

extern const std::vector<Loc> LOCS;

extern const std::vector<std::string> LOC_STRS;

extern const std::vector<std::string> VALID_LOC_STRS;

extern const std::unordered_map<std::string, Loc> LOC_FROM_STR;

extern const std::unordered_map<Loc, size_t> LOC_IDX;

extern const std::vector<Loc> ONLY_COAST_LOCS;

// Return true if loc is open water
bool is_water(Loc);

// Return true if loc is a coastal land loc
bool is_coast(Loc);

// Return true if loc is a supply center
bool is_center(Loc);

// Return the string representation of a loc
std::string loc_str(Loc);
Loc loc_from_str(const std::string &);

// Map BUL/EC -> BUL, non-coasts to themselves
Loc root_loc(Loc);

// Map any of BUL, BUL/EC, BUL/SC -> {BUL, BUL/EC, BUL/SC}
const std::vector<Loc> &expand_coasts(Loc);

// Return a vector of root home supply centers for the given power
const std::vector<Loc> &home_centers(Power power);

// Return a vector of locs where armies can be built
const std::vector<Loc> &home_centers_army(Power power);

// Return a vector of locs where fleets can be built
const std::vector<Loc> &home_centers_fleet(Power power);

// Print operator
std::ostream &operator<<(std::ostream &os, Loc loc);

NLOHMANN_JSON_SERIALIZE_ENUM(
    Loc,
    {{Loc::NONE, "NONE"},     {Loc::YOR, "YOR"},       {Loc::EDI, "EDI"},
     {Loc::LON, "LON"},       {Loc::LVP, "LVP"},       {Loc::NTH, "NTH"},
     {Loc::WAL, "WAL"},       {Loc::CLY, "CLY"},       {Loc::NWG, "NWG"},
     {Loc::ENG, "ENG"},       {Loc::IRI, "IRI"},       {Loc::NAO, "NAO"},
     {Loc::BEL, "BEL"},       {Loc::DEN, "DEN"},       {Loc::HEL, "HEL"},
     {Loc::HOL, "HOL"},       {Loc::NWY, "NWY"},       {Loc::SKA, "SKA"},
     {Loc::BAR, "BAR"},       {Loc::BRE, "BRE"},       {Loc::MAO, "MAO"},
     {Loc::PIC, "PIC"},       {Loc::BUR, "BUR"},       {Loc::RUH, "RUH"},
     {Loc::BAL, "BAL"},       {Loc::KIE, "KIE"},       {Loc::SWE, "SWE"},
     {Loc::FIN, "FIN"},       {Loc::STP, "STP"},       {Loc::STP_NC, "STP/NC"},
     {Loc::GAS, "GAS"},       {Loc::PAR, "PAR"},       {Loc::NAF, "NAF"},
     {Loc::POR, "POR"},       {Loc::SPA, "SPA"},       {Loc::SPA_NC, "SPA/NC"},
     {Loc::SPA_SC, "SPA/SC"}, {Loc::WES, "WES"},       {Loc::MAR, "MAR"},
     {Loc::MUN, "MUN"},       {Loc::BER, "BER"},       {Loc::BOT, "BOT"},
     {Loc::LVN, "LVN"},       {Loc::PRU, "PRU"},       {Loc::STP_SC, "STP/SC"},
     {Loc::MOS, "MOS"},       {Loc::TUN, "TUN"},       {Loc::LYO, "LYO"},
     {Loc::TYS, "TYS"},       {Loc::PIE, "PIE"},       {Loc::BOH, "BOH"},
     {Loc::SIL, "SIL"},       {Loc::TYR, "TYR"},       {Loc::WAR, "WAR"},
     {Loc::SEV, "SEV"},       {Loc::UKR, "UKR"},       {Loc::ION, "ION"},
     {Loc::TUS, "TUS"},       {Loc::NAP, "NAP"},       {Loc::ROM, "ROM"},
     {Loc::VEN, "VEN"},       {Loc::GAL, "GAL"},       {Loc::VIE, "VIE"},
     {Loc::TRI, "TRI"},       {Loc::ARM, "ARM"},       {Loc::BLA, "BLA"},
     {Loc::RUM, "RUM"},       {Loc::ADR, "ADR"},       {Loc::AEG, "AEG"},
     {Loc::ALB, "ALB"},       {Loc::APU, "APU"},       {Loc::EAS, "EAS"},
     {Loc::GRE, "GRE"},       {Loc::BUD, "BUD"},       {Loc::SER, "SER"},
     {Loc::ANK, "ANK"},       {Loc::SMY, "SMY"},       {Loc::SYR, "SYR"},
     {Loc::BUL, "BUL"},       {Loc::BUL_EC, "BUL/EC"}, {Loc::CON, "CON"},
     {Loc::BUL_SC, "BUL/SC"}})

// Number of supply-centers in the game
static const uint64_t N_SCS = 34;

} // namespace dipcc
