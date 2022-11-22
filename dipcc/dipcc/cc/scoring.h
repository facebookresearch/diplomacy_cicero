/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <array>
#include <optional>
#include <string>

enum class Scoring {
  SOS = 0, // sum of squares. In case of non-solo, score is proportional to
           // centers^2
  DSS = 1, // draw size scoring. In case of non-solo, score is equal among
           // surviving players
};

const std::array<std::string, 2> SCORING_STRINGS{"sum_of_squares", "draw_size"};

constexpr int NUM_SCORING_SYSTEMS = SCORING_STRINGS.size();

std::optional<Scoring> scoring_from_string(const std::string &s);
