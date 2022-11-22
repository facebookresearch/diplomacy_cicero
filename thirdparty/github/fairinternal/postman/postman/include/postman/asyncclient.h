/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <future>

#include <ATen/ATen.h>
#include <grpc++/grpc++.h>
#include <nest.h>

#include "rpc.grpc.pb.h"

typedef nest::Nest<at::Tensor> TensorNest;

namespace postman {
class AsyncClient {
 public:
  class Streams {
    template <typename T>
    class Queue {
     public:
      bool push(T* item) {
        std::unique_lock<std::mutex> lock(mu_);
        if (closed()) {
          return false;
        }
        deque_.push_back(std::unique_ptr<T>(item));
        return true;
      }

      std::unique_ptr<T> pop() {
        std::unique_ptr<T> result;
        std::unique_lock<std::mutex> lock(mu_);
        if (!closed() && !deque_.empty()) {
          result = std::move(deque_.front());
          deque_.pop_front();
        }
        return result;
      }

      std::deque<std::unique_ptr<T>>& close() {
        std::unique_lock<std::mutex> lock(mu_);
        closed_.store(true);
        return deque_;
      }

      bool closed() const { return closed_.load(); }

     private:
      std::mutex mu_;
      std::atomic_bool closed_ = false;
      std::deque<std::unique_ptr<T>> deque_;
    };

    /// Async gRPC idiom, see
    ///   https://grpc.io/docs/tutorials/async/helloasync-cpp/
    class CallData {
     public:
      CallData(RPC::Stub* stub, grpc::CompletionQueue* cq,
               Queue<CallData>* queue, std::promise<grpc::Status> result)
          : stream_(stub->PrepareAsyncCall(&context_, cq)),
            queue_(queue),
            result_(std::move(result)) {}

      std::future<TensorNest> call(const std::string& function,
                                   const TensorNest& inputs);

      void finish() {
        GPR_ASSERT(status_ == PROCESS);

        // We cannot simply call Finish() it seems, the
        // server will only wake up from Read() by WritesDone().
        // This isn't quite clear from the gRPC docs for
        // grpc_impl::internal::ClientAsyncStreamingInterface::Finish.
        status_ = WRITES_DONE;
        stream_->WritesDone(this);
      }

      void proceed();

     private:
      grpc::ClientContext context_;
      std::unique_ptr<grpc::ClientAsyncReaderWriterInterface<
          postman::CallRequest, postman::CallResponse>>
          stream_;
      Queue<CallData>* queue_;

      enum Status { CREATE, PROCESS, WRITE, READ, WRITES_DONE, FINISH };
      Status status_ = CREATE;

      CallRequest request_;
      CallResponse response_;

      std::promise<TensorNest> promise_;
      std::promise<grpc::Status> result_;
      grpc::Status result_value_;
    };

   public:
    Streams(RPC::Stub* stub);

    ~Streams();

    /// Make an async call.
    ///
    /// \param function Name of the function to call.
    /// \param inputs The inputs of the function to call.
    ///
    /// This method will start a new (bidi streaming) call when
    /// none is available and re-use existing calls when they are.
    /// Calls undergo a lifecycle of CREATE->PROCESS->WRITE->READ.
    /// From READ they can go back into PROCESS or into
    /// WRITES_DONE->FINISH. From PROCESS (idle) they can also go
    /// into WRITES_DONE. See CallData class above for details.
    ///
    /// TODO(heiner): Consider limiting the number of parallel calls,
    /// blocking (or failing?) when none are available.
    ///
    /// \return A future for the return value.
    std::future<TensorNest> call(const std::string& function,
                                 const TensorNest& inputs);

    void close();

   private:
    RPC::Stub* stub_;
    grpc::CompletionQueue cq_;

    std::unique_ptr<std::thread> polling_thread_;
    Queue<CallData> queue_;
    std::vector<std::future<grpc::Status>> stati_;
  };

  AsyncClient(const std::string& address) : address_(address) {}

  std::shared_ptr<AsyncClient::Streams> connect(int deadline_sec = 60);

 private:
  const std::string address_;
  std::unique_ptr<RPC::Stub> stub_;
};  // namespace postman

}  // namespace postman
