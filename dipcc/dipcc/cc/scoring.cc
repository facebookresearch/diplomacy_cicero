/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "scoring.h"

std::optional<Scoring> scoring_from_string(const std::string &s) {
  for (size_t i = 0; i < NUM_SCORING_SYSTEMS; ++i) {
    if (s == SCORING_STRINGS[i]) {
      return static_cast<Scoring>(i);
    }
  }
  return {};
}
