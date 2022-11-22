/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <exception>
#include <stdexcept>
#include <string>

#include "checks.h"

namespace dipcc {

void JCHECK(bool b, const std::string &msg) {
  if (!b) {
    throw std::runtime_error(msg);
  }
}

void JCHECK(bool b) {
  if (!b) {
    throw std::runtime_error("JCHECK failed");
  }
}

[[noreturn]] void JFAIL(const std::string &msg) {
  throw std::runtime_error(msg);
}

[[noreturn]] void JFAIL() { throw std::runtime_error("JCHECK failed"); }

} // namespace dipcc
