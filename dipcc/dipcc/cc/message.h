/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "phase.h"
#include "power.h"
#include <string>

namespace dipcc {

struct Message {
  Power sender;
  PowerOrAll recipient;
  Phase phase;
  std::string message;
  uint64_t time_sent;
};

} // namespace dipcc
