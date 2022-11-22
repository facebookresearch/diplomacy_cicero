/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <nest.h>

#include <ATen/ATen.h>

#include "rpc.pb.h"

typedef nest::Nest<at::Tensor> TensorNest;

namespace postman {
namespace detail {
// Fill an ArrayNest proto from a TensorNest.
void fill_proto_from_tensornest(postman::ArrayNest* nest_pb,
                                const TensorNest& nest);

// Create a TensorNest from an ArrayNest proto.
TensorNest nest_proto_to_tensornest(postman::ArrayNest* nest_pb);
}  // namespace detail
}  // namespace postman
