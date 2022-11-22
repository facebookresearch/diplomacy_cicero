/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <glog/logging.h>

#include "checks.h"
#include "phase.h"

namespace dipcc {

Phase::Phase(char season, uint32_t year, char phase_type) {
  this->season = season;
  this->year = year;
  this->phase_type = phase_type;
}

Phase::Phase(const std::string &s) {
  if (s == "COMPLETED") {
    this->season = 'C';
    this->year = 2000;
    this->phase_type = 'C';
  } else {
    JCHECK(s.size() == 6, "Bad phase string: " + s);
    this->season = s.at(0);
    this->year = std::stoi(s.substr(1, 4));
    this->phase_type = s.at(5);
  }
}

Phase Phase::next(bool retreat) const {
  if (retreat) {
    JCHECK(this->phase_type == 'M', "R phase must follow M phase");
    return Phase(this->season, this->year, 'R');
  }
  if (this->season == 'S') {
    return Phase('F', this->year, 'M');
  }
  if (this->season == 'F') {
    return Phase('W', this->year, 'A');
  }
  if (this->season == 'W') {
    return Phase('S', this->year + 1, 'M');
  }
  JFAIL("next() called on bad phase: " + this->to_string());
}

Phase Phase::completed() const { return {'C', this->year, 'C'}; }

std::string Phase::to_string() const {
  if (this->phase_type == 'C') {
    return "COMPLETED";
  }

  char buf[7];
  snprintf(buf, sizeof(buf), "%c%d%c", season, year, phase_type);
  return std::string(buf);
}

std::string Phase::to_string_long() const {
  if (this->phase_type == 'C') {
    return "COMPLETED";
  }

  std::string s;

  if (this->season == 'S') {
    s.append("SPRING ");
  } else if (this->season == 'F') {
    s.append("FALL ");
  } else if (this->season == 'W') {
    s.append("WINTER ");
  }

  s.append(std::to_string(this->year));

  if (this->phase_type == 'M') {
    s.append(" MOVEMENT");
  } else if (this->phase_type == 'R') {
    s.append(" RETREATS");
  } else if (this->phase_type == 'A') {
    s.append(" ADJUSTMENTS");
  }

  return s;
}

bool Phase::operator<(const Phase &other) const {
  if (this->season == 'C') {
    return false;
  }
  if (other.season == 'C') {
    return true;
  }
  if (this->year != other.year) {
    return this->year < other.year;
  }
  if (this->season != other.season) {
    for (char s : {'S', 'F', 'W'}) {
      if (this->season == s) {
        return true;
      } else if (other.season == s) {
        return false;
      }
    }
  }
  if (this->phase_type != other.phase_type) {
    for (char s : {'M', 'R', 'A'}) {
      if (this->phase_type == s) {
        return true;
      } else if (other.phase_type == s) {
        return false;
      }
    }
  }

  // they are equal
  return false;
}

bool Phase::operator==(const Phase &other) const {
  if (this->season == 'C' || other.season == 'C') {
    return this->season == other.season;
  }
  return (this->season == other.season) && (this->year == other.year) &&
         (this->phase_type == other.phase_type);
}

bool Phase::operator!=(const Phase &other) const { return !(*this == other); }
bool Phase::operator>(const Phase &other) const { return other < *this; }
bool Phase::operator<=(const Phase &other) const { return !(*this > other); }
bool Phase::operator>=(const Phase &other) const { return !(*this < other); }

} // namespace dipcc
