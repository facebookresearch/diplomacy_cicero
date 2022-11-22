#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import threading
import unittest

import numpy as np

import torch

import postman


class AsyncRPCTest(unittest.TestCase):
    def test_simple(self, address="127.0.0.1:12346"):
        def function(a, b, c):
            return a + 1, b + 2, c + 3

        server = postman.Server(address)
        server.bind("function", function, batch_size=1)
        server.run()

        try:
            client = postman.AsyncClient(address)
            streams = client.connect(10)

            inputs = (torch.zeros(1), torch.ones(2), torch.arange(10))

            future = streams.function(*inputs)

            result = future.get()
            for x, y in zip(result, function(*inputs)):
                np.testing.assert_array_equal(x, y)

        finally:
            streams.close()
            server.stop()

    def test_abba(self, address="127.0.0.1:12346"):
        event = threading.Event()

        def a(x):
            event.wait()
            return x + 1

        def b(x):
            return x + 2

        server = postman.Server(address)
        server.bind("a", a, batch_size=1)
        server.bind("b", b, batch_size=1)
        server.run()

        try:
            client = postman.AsyncClient(address)
            streams = client.connect(10)

            a_future = streams.a(torch.zeros(()))
            b_future = streams.b(torch.zeros(()))

            event.set()

            a_result = a_future.get()
            b_result = b_future.get()

            np.testing.assert_array_equal(a_result, torch.full((), 1))
            np.testing.assert_array_equal(b_result, torch.full((), 2))
        finally:
            streams.close()
            server.stop()


if __name__ == "__main__":
    unittest.main()
