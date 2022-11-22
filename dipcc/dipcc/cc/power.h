/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>
#include <vector>

#include "thirdparty/nlohmann/json.hpp"

namespace dipcc {

enum class Power {
  NONE,
  AUSTRIA,
  ENGLAND,
  FRANCE,
  GERMANY,
  ITALY,
  RUSSIA,
  TURKEY,
};
enum class PowerOrAll {
  NONE,
  AUSTRIA,
  ENGLAND,
  FRANCE,
  GERMANY,
  ITALY,
  RUSSIA,
  TURKEY,
  ALL,
};

static constexpr int NUM_POWERS = 7;
const Power POWERS[NUM_POWERS] = {Power::AUSTRIA, Power::ENGLAND, Power::FRANCE,
                                  Power::GERMANY, Power::ITALY,   Power::RUSSIA,
                                  Power::TURKEY};
static constexpr int NUM_POWERS_OR_ALL = 8;
const PowerOrAll POWERS_OR_ALL[NUM_POWERS_OR_ALL] = {
    PowerOrAll::AUSTRIA, PowerOrAll::ENGLAND, PowerOrAll::FRANCE,
    PowerOrAll::GERMANY, PowerOrAll::ITALY,   PowerOrAll::RUSSIA,
    PowerOrAll::TURKEY,  PowerOrAll::ALL};

const std::vector<std::string> POWERS_STR{
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
};
const std::vector<std::string> POWERS_OR_ALL_STR{
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY",
    "ITALY",   "RUSSIA",  "TURKEY", "ALL",
};

std::string power_str(const Power &power);
std::string power_or_all_str(const PowerOrAll &power_or_all);
Power power_from_str(const std::string &s);
PowerOrAll power_or_all_from_str(const std::string &s);

NLOHMANN_JSON_SERIALIZE_ENUM(Power, {{Power::AUSTRIA, "AUSTRIA"},
                                     {Power::ENGLAND, "ENGLAND"},
                                     {Power::FRANCE, "FRANCE"},
                                     {Power::GERMANY, "GERMANY"},
                                     {Power::ITALY, "ITALY"},
                                     {Power::RUSSIA, "RUSSIA"},
                                     {Power::TURKEY, "TURKEY"}})

NLOHMANN_JSON_SERIALIZE_ENUM(PowerOrAll, {{PowerOrAll::AUSTRIA, "AUSTRIA"},
                                          {PowerOrAll::ENGLAND, "ENGLAND"},
                                          {PowerOrAll::FRANCE, "FRANCE"},
                                          {PowerOrAll::GERMANY, "GERMANY"},
                                          {PowerOrAll::ITALY, "ITALY"},
                                          {PowerOrAll::RUSSIA, "RUSSIA"},
                                          {PowerOrAll::TURKEY, "TURKEY"},
                                          {PowerOrAll::ALL, "ALL"}})

} // namespace dipcc
