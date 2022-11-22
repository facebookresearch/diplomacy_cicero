/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <ATen/ATen.h>
#include <grpc++/grpc++.h>
#include <nest.h>

#include "exceptions.h"
#include "rpc.grpc.pb.h"

typedef nest::Nest<at::Tensor> TensorNest;

namespace postman {
class Client {
 public:
  Client(const std::string& address) : address_(address) {}

  void connect(int deadline_sec = 60);

  TensorNest call(const std::string& function, const TensorNest& inputs);

 private:
  const std::string address_;
  std::unique_ptr<RPC::Stub> stub_;
  grpc::ClientContext context_;
  std::shared_ptr<grpc::ClientReaderWriter<CallRequest, CallResponse>> stream_;
};

}  // namespace postman
