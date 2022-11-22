/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "thirdparty/nlohmann/json.hpp"

using nlohmann::json;

namespace dipcc {

void to_json(json &j, const Unit &x);
void to_json(json &j, const OwnedUnit &x);
void to_json(json &j, const Phase &x);
void to_json(json &j, const GameState &x);
void to_json(json &j, const Message &x);
void from_json(const json &j, Message &x);

} // namespace dipcc
