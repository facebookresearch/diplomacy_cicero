/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <exception>
#include <string>

namespace dipcc {

void JCHECK(bool b, const std::string &msg);
void JCHECK(bool b);

[[noreturn]] void JFAIL(const std::string &msg);
[[noreturn]] void JFAIL();

} // namespace dipcc
