/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <memory>

#include "postman/client.h"
#include "postman/computationqueue.h"
#include "postman/server.h"

TEST(ServerClientTest, ViaBind) {
  static std::string address = "127.0.0.1:54321";

  postman::Server server(address);
  server.bind("myfunction", [&](const TensorNest& inputs) {
    return inputs.map([](at::Tensor t) { return t + 7; });
  });

  server.run();

  postman::Client client(address);

  TensorNest inputs(at::zeros(1));

  ASSERT_THROW(client.call("myfunction", inputs), postman::ConnectionError);

  client.connect(3);

  std::cerr << "Testing an unknown function. Expect a log statement: ";
  ASSERT_THROW(client.call("doesntexist", inputs), postman::CallError);

  TensorNest outputs = client.call("myfunction", inputs);

  TensorNest::for_each(
      [](at::Tensor t1, at::Tensor t2) { ASSERT_TRUE(at::equal(t1 + 7, t2)); },
      inputs, outputs);

  server.stop();
}

TEST(ServerClientTest, ViaComputationQueue) {
  static std::string address = "127.0.0.1:54321";

  postman::Server server(address);
  auto queue = std::make_shared<postman::ComputationQueue>(1);

  server.bind_queue("myfunction", queue);

  std::thread queue_read([&]() {
    try {
      auto computation = queue->get(true);
      TensorNest inputs = computation->get_inputs();
      TensorNest outputs = inputs.map([](at::Tensor t) { return t + 42; });
      computation->set_outputs(outputs);
    } catch (const postman::QueueClosed& e) {
    }
  });

  server.run();

  postman::Client client(address);

  TensorNest inputs(at::zeros(1));

  ASSERT_THROW(client.call("myfunction", inputs), postman::ConnectionError);

  client.connect(3);

  std::cerr << "Testing an unknown function. Expect a log statement: ";
  ASSERT_THROW(client.call("doesntexist", inputs), postman::CallError);

  TensorNest outputs = client.call("myfunction", inputs);

  TensorNest::for_each(
      [](at::Tensor t1, at::Tensor t2) { ASSERT_TRUE(at::equal(t1 + 42, t2)); },
      inputs, outputs);

  queue->close();
  server.stop();
  queue_read.join();
}
