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

#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include <ATen/ATen.h>

#include "postman/serialization.h"
#include "postman/server.h"

#include "postman/blocking_counter.h"
#include "postman/computationqueue.h"
#include "postman/exceptions.h"

#include <nest.h>

namespace postman {

grpc::Status Server::ServiceImpl::bind(const std::string &name,
                                       Function &&function) {
  functions_.insert({name, std::move(function)});
  return grpc::Status::OK;
}

grpc::Status Server::ServiceImpl::Call(
    grpc::ServerContext *context,
    grpc::ServerReaderWriter<CallResponse, CallRequest> *stream) {
  CallRequest call_req;

  while (stream->Read(&call_req)) {
    CallResponse call_resp;
    try {
      auto it = functions_.find(call_req.function());
      if (it == functions_.end())
        throw std::runtime_error("AttributeError: No such function '" +
                                 call_req.function() + "'");
      TensorNest result = it->second(
          detail::nest_proto_to_tensornest(call_req.mutable_inputs()));
      detail::fill_proto_from_tensornest(call_resp.mutable_outputs(), result);
    } catch (const QueueClosed &e) {
      break;
    } catch (const std::runtime_error &e) {
      std::cerr << "Error in " << call_req.function() << ": " << e.what()
                << std::endl;
      call_resp.mutable_error()->set_message(e.what());
    } catch (const std::exception &e) {
      std::cerr << "Error in " << call_req.function() << ": " << e.what()
                << std::endl;
      return grpc::Status(grpc::INTERNAL, e.what());
    }
    stream->Write(call_resp);
  }

  return grpc::Status::OK;
}

void Server::run() {
  if (server_) throw std::runtime_error("Server already running");

  int port;

  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(-1);  // Unlimited.
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&service_);
  server_ = builder.BuildAndStart();

  if (!server_)
    throw std::runtime_error(
        "Failed to run server. Maybe the port is already used?");

  if (port == 0) throw std::runtime_error("Failed to bind to port");

  port_.store(port);
  running_.store(true);
}

void Server::wait() {
  if (!server_) throw std::runtime_error("Server not running");

  server_->Wait();
}

void Server::stop() {
  if (!server_) throw std::runtime_error("Server not running");

  running_.store(false);
  server_->Shutdown(std::chrono::system_clock::now());
}

void Server::bind(const std::string &name, Function &&function) {
  service_.bind(name, std::move(function));
}

void Server::bind_queue(const std::string &name,
                        std::shared_ptr<ComputationQueue> queue) {
  bind(name, [queue(queue)](const TensorNest &inputs) mutable {
    int64_t index;
    auto future = queue->compute(inputs, &index);

    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
      throw TimeoutError("Compute timeout reached.");

    TensorNest outputs = [&]() {
      try {
        return future.get();
      } catch (const std::future_error &e) {
        if (queue->closed() && e.code() == std::future_errc::broken_promise)
          throw QueueClosed(e.what());
        throw;
      }
    }();

    return outputs.map([index](const at::Tensor &t) { return t[index]; });
  });
}

void Server::bind_queue_batched(const std::string &name,
                                std::shared_ptr<ComputationQueue> queue) {
  bind(name, [queue(queue)](const TensorNest &inputs) mutable {
    // TODO: Add some shape testing.
    int64_t batch_size = inputs.front().size(0);

    std::vector<int64_t> indices(batch_size);
    std::vector<std::shared_future<TensorNest>> futures;

    for (int64_t i = 0; i < batch_size; ++i) {
      futures.push_back(queue->compute(
          inputs.map([i](const at::Tensor &t) { return t[i]; }), &indices[i]));
    }

    std::vector<TensorNest> outputs;

    for (int64_t i = 0; i < batch_size; ++i) {
      try {
        outputs.push_back(futures[i].get().map(
            [index(indices[i])](const at::Tensor &t) { return t[index]; }));
      } catch (const std::future_error &e) {
        if (queue->closed() && e.code() == std::future_errc::broken_promise)
          throw QueueClosed(e.what());
        throw;
      }
    }

    // TODO: This incurs extra memcopies. We could also write everything to a
    // buffer here and give that to the protobuf library directly.
    nest::Nest<std::vector<at::Tensor>> zipped = TensorNest::zip(outputs);
    return zipped.map(
        [](const std::vector<at::Tensor> &v) { return at::stack(v); });
  });
}

}  // namespace postman
