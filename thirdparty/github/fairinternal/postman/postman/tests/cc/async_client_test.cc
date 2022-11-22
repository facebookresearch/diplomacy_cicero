/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <memory>

#include "postman/asyncclient.h"
#include "postman/computationqueue.h"
#include "postman/exceptions.h"
#include "postman/server.h"

TEST(AsyncClientTest, Simple) {
  static std::string address = "127.0.0.1:54324";

  postman::Server server(address);
  server.bind("myfunction", [&](const TensorNest& inputs) {
    return inputs.map([](at::Tensor t) { return t + 7; });
  });

  server.run();

  postman::AsyncClient client(address);

  TensorNest inputs(at::zeros(1));

  std::shared_ptr<postman::AsyncClient::Streams> streams = client.connect(3);

  std::cerr << "Testing an unknown function. Expect a log statement: ";
  std::future<TensorNest> future = streams->call("doesntexist", inputs);
  ASSERT_THROW(future.get(), postman::CallError);

  future = streams->call("myfunction", inputs);

  TensorNest outputs = future.get();

  TensorNest::for_each(
      [](at::Tensor t1, at::Tensor t2) {
        std::cerr << t2 << std::endl;
        ASSERT_TRUE(at::equal(t1 + 7, t2));
      },
      inputs, outputs);

  streams->close();
  server.stop();
}

TEST(AsyncClientTest, TestABBA) {
  static std::string address = "127.0.0.1:54325";

  postman::Server server(address);

  std::promise<void> promise;
  auto future = promise.get_future();

  server.bind("a", [&](const TensorNest& inputs) {
    future.get();
    return inputs.map([](at::Tensor t) { return t + 1; });
  });

  server.bind("b", [&](const TensorNest& inputs) {
    return inputs.map([](at::Tensor t) { return t + 2; });
  });

  server.run();

  postman::AsyncClient client(address);

  TensorNest inputs(at::zeros(1));

  std::shared_ptr<postman::AsyncClient::Streams> streams = client.connect(3);

  auto a_future = streams->call("a", inputs);
  auto b_future = streams->call("b", inputs);

  promise.set_value();

  TensorNest a = a_future.get();
  TensorNest b = b_future.get();

  ASSERT_TRUE(at::equal(a.front(), at::full(1, 1)));
  ASSERT_TRUE(at::equal(b.front(), at::full(1, 2)));

  streams->close();
  server.stop();
}

TEST(AsyncClientTest, TestNoGetFutureForYou) {
  static std::string address = "127.0.0.1:54327";

  postman::Server server(address);
  server.bind("myfunction", [&](const TensorNest& inputs) { return inputs; });

  server.run();

  postman::AsyncClient client(address);

  TensorNest inputs(at::zeros(1));

  std::shared_ptr<postman::AsyncClient::Streams> streams = client.connect(3);

  std::future<TensorNest> future = streams->call("myfunction", inputs);

  streams->close();
  server.stop();
}
