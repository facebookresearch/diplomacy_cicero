/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

namespace postman {

class ConnectionError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
class TimeoutError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
class CallError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
}  // namespace postman
