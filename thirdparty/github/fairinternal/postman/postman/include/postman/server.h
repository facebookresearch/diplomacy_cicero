/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <atomic>

#include <grpc++/grpc++.h>

#include "rpc.grpc.pb.h"
#include "rpc.pb.h"

#include "computationqueue.h"

namespace postman {
class Server {
  typedef std::function<TensorNest(const TensorNest &)> Function;

  class ServiceImpl final : public RPC::Service {
   public:
    grpc::Status bind(const std::string &name, Function &&function);

   private:
    virtual grpc::Status Call(
        grpc::ServerContext *context,
        grpc::ServerReaderWriter<CallResponse, CallRequest> *stream) override;

    std::map<std::string, Function> functions_;
  };

 public:
  Server(const std::string &address) : address_(address), server_(nullptr) {}

  void run();
  void wait();
  void stop();

  bool running() { return running_.load(); }
  int port() { return port_.load(); }

  void bind(const std::string &name, Function &&function);

  void bind_queue(const std::string &name,
                  std::shared_ptr<ComputationQueue> queue);
  void bind_queue_batched(const std::string &name,
                          std::shared_ptr<ComputationQueue> queue);

 private:
  const std::string address_;
  ServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;

  std::atomic_bool running_ = false;
  std::atomic_int port_ = 0;
};

}  // namespace postman
