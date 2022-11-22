/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <string>

#include "enums.h"

namespace dipcc {

struct Phase {
  char season; // 'S', 'F', 'W', 'C'
  uint32_t year;
  char phase_type; // 'M', 'R', 'A', 'C'

  Phase() {}
  Phase(char, uint32_t, char);
  Phase(const std::string &s);

  Phase next(bool retreat) const;

  Phase completed() const;

  std::string to_string() const;
  std::string to_string_long() const;

  bool operator<(const Phase &other) const;
  bool operator>(const Phase &other) const;
  bool operator<=(const Phase &other) const;
  bool operator>=(const Phase &other) const;
  bool operator==(const Phase &other) const;
  bool operator!=(const Phase &other) const;
};

void to_json(json &j, const Phase &x);

} // namespace dipcc
