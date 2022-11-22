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

#include "postman/asyncclient.h"
#include "postman/exceptions.h"
#include "postman/serialization.h"

namespace postman {

std::future<TensorNest> AsyncClient::Streams::CallData::call(
    const std::string& function, const TensorNest& inputs) {
  GPR_ASSERT(status_ == PROCESS || status_ == CREATE);
  promise_ = std::promise<TensorNest>();
  auto future = promise_.get_future();

  request_.Clear();
  request_.set_function(function);
  detail::fill_proto_from_tensornest(request_.mutable_inputs(), inputs);

  proceed();
  return future;
}

void AsyncClient::Streams::CallData::proceed() {
  switch (status_) {
    case CREATE:
      status_ = PROCESS;
      stream_->StartCall(this);
      return;
    case PROCESS:
      status_ = WRITE;
      stream_->Write(request_, this);
      return;
    case WRITE:
      status_ = READ;
      stream_->Read(&response_, this);
      return;
    case READ:
      if (response_.has_error()) {
        promise_.set_exception(
            std::make_exception_ptr(CallError(response_.error().message())));
      } else {
        promise_.set_value(
            detail::nest_proto_to_tensornest(response_.mutable_outputs()));
      }
      if (queue_->push(this))
        status_ = PROCESS;
      else {
        status_ = WRITES_DONE;
        stream_->WritesDone(this);
      }
      return;
    case WRITES_DONE:
      status_ = FINISH;
      stream_->Finish(&result_value_, this);
      return;
    case FINISH:
      result_.set_value(result_value_);
      delete this;
  }
}

AsyncClient::Streams::Streams(RPC::Stub* stub) : stub_(stub) {
  // TODO(heiner): Consider using more threads on request.
  polling_thread_ = std::make_unique<std::thread>(([this]() {
    void* untyped_tag;
    bool ok;
    while (cq_.Next(&untyped_tag, &ok)) {
      CallData* tag = static_cast<CallData*>(untyped_tag);
      if (ok) {
        tag->proceed();
      } else {
        delete tag;
      }
    }
  }));
}

AsyncClient::Streams::~Streams() {
  if (polling_thread_) {
    std::cerr << "Warning: Streams object wasn't closed before destruction."
              << std::endl;
    close();
  }
}

std::future<TensorNest> AsyncClient::Streams::call(const std::string& function,
                                                   const TensorNest& inputs) {
  std::unique_ptr<CallData> calldata = queue_.pop();
  if (!calldata) {  // Queue was empty or closed.
    if (queue_.closed()) throw ConnectionError("Streams are closed");
    std::promise<grpc::Status> promise;
    stati_.push_back(promise.get_future());
    calldata.reset(new CallData(stub_, &cq_, &queue_, std::move(promise)));
  }
  return calldata.release()->call(function, inputs);
}

void AsyncClient::Streams::close() {
  std::deque<std::unique_ptr<CallData>> deque = std::move(queue_.close());
  for (auto& calldata : deque) {
    calldata.release()->finish();
  }

  for (auto& future : stati_) {
    future.wait();
  }

  // All CallData objects finished now.
  cq_.Shutdown();
  polling_thread_->join();
  polling_thread_.reset();

  for (auto& future : stati_) {
    grpc::Status status = future.get();
    if (!status.ok()) {
      throw ConnectionError(status.error_message() + " (" +
                            std::to_string(status.error_code()) + ")");
    }
  }
}

std::shared_ptr<AsyncClient::Streams> AsyncClient::connect(int deadline_sec) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(address_, grpc::InsecureChannelCredentials());
  stub_ = RPC::NewStub(channel);

  auto deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(deadline_sec);

  if (!channel->WaitForConnected(deadline)) {
    throw TimeoutError("WaitForConnected timed out.");
  }

  return std::make_shared<AsyncClient::Streams>(stub_.get());
}

}  // namespace postman
