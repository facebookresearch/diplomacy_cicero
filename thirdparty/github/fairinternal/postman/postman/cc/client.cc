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

#include <iostream>
#include <tuple>

#include "postman/client.h"
#include "postman/serialization.h"

namespace postman {
void Client::connect(int deadline_sec) {
  grpc::ChannelArguments ch_args;
  ch_args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateCustomChannel(address_, grpc::InsecureChannelCredentials(), ch_args);
  stub_ = RPC::NewStub(channel);

  auto deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(deadline_sec);

  if (!channel->WaitForConnected(deadline)) {
    throw TimeoutError("WaitForConnected timed out.");
  }
  stream_ = stub_->Call(&context_);
}

TensorNest Client::call(const std::string& function, const TensorNest& inputs) {
  if (!stream_) throw ConnectionError("Client not connected");

  postman::CallRequest call_req;
  call_req.set_function(function);
  detail::fill_proto_from_tensornest(call_req.mutable_inputs(), inputs);

  CallResponse call_resp;
  try {
    if (!stream_->Write(call_req)) throw std::runtime_error("Write failed");
    if (!stream_->Read(&call_resp)) throw std::runtime_error("Read failed");
  } catch (const std::runtime_error& e) {
    grpc::Status status = stream_->Finish();
    if (status.ok()) {
      throw ConnectionError("Server closed stream");
    }
    throw ConnectionError(std::string(e.what()) + ": " +
                          status.error_message() + " (" +
                          std::to_string(status.error_code()) + ")");
  }

  if (call_resp.has_error()) throw CallError(call_resp.error().message());

  return detail::nest_proto_to_tensornest(call_resp.mutable_outputs());
}

}  // namespace postman
