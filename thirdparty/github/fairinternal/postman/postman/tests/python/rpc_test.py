#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import unittest
import multiprocessing as mp

import numpy as np

import torch

import postman


class RPCTest(unittest.TestCase):
    def test_rpc_python(self, num_clients=2, address="127.0.0.1:12346"):
        def run_client():
            client = postman.Client(address)
            client.connect(10)
            client.py_function(
                torch.zeros((1, 2)), torch.arange(10), (torch.empty(2, 3), torch.ones((1, 2))),
            )
            client.batched_function(torch.zeros((1, 2)))

        client_processes = [mp.Process(target=run_client) for _ in range(num_clients)]

        calls = collections.defaultdict(int)

        def py_function(a, b, c):
            calls["py_function"] += 1
            np.testing.assert_array_equal(a.numpy(), np.zeros((1, 1, 2)))
            np.testing.assert_array_equal(b.numpy(), np.arange(10).reshape((1, 10)))

            c0, c1 = c
            self.assertSequenceEqual(list(c0.shape), (1, 2, 3))
            np.testing.assert_array_equal(c1.numpy(), np.ones((1, 1, 2)))

            return torch.ones(1, 1)

        def batched_function(a):
            calls["batched_function"] += 1
            self.assertEqual(a.shape[0], 2)
            return torch.ones(a.shape)

        server = postman.Server(address)
        server.bind("py_function", py_function, batch_size=1)
        server.bind(
            "batched_function", batched_function, batch_size=num_clients, wait_till_full=True,
        )
        server.run()

        for p in client_processes:
            p.start()

        for p in client_processes:
            p.join()

        server.stop()

        self.assertEqual(calls["py_function"], num_clients)
        self.assertEqual(calls["batched_function"], 1)

    @unittest.skip("disabled until jit is re-added")
    def test_rpc_jit(self, num_clients=2, address="127.0.0.1:12346"):
        def run_client(client_id):
            client = postman.Client(address)
            client.connect(10)
            arg = np.full((1, 2), client_id, dtype=np.float32)
            batched_arg = np.full((2,), client_id, dtype=np.float32)

            function_result = client.function(arg)
            batched_function_result = client.batched_function(batched_arg)

            np.testing.assert_array_equal(function_result, np.full((1, 2), client_id))
            np.testing.assert_array_equal(batched_function_result, np.full((2,), client_id))

        clients = [mp.Process(target=run_client, args=(i,)) for i in range(num_clients)]

        linear = torch.nn.Linear(2, 2, bias=False)
        linear.weight.data = torch.diagflat(torch.ones(2))
        module = torch.jit.script(linear)
        server = postman.Server("127.0.0.1:12346")

        server.bind("function", module)
        server.bind("batched_function", module, batch_size=num_clients)

        server.run()

        for p in clients:
            p.start()

        for p in clients:
            p.join()

        server.stop()

    # TODO(heiner): Add more tests: return values, etc.

    def test_none_return(self):
        def get_nothing():
            # TODO(heiner): Add check on return shape.
            return torch.arange(2).reshape(1, 2)

        def return_nothing(t):
            return None

        def nothing():
            return

        server = postman.Server("127.0.0.1:0")
        server.bind("get_nothing", get_nothing, batch_size=1)
        server.bind("return_nothing", return_nothing, batch_size=1)
        server.bind("nothing", nothing, batch_size=1)
        server.run()

        client = postman.Client("127.0.0.1:%i" % server.port())
        client.connect(10)
        try:
            value = client.get_nothing()
            np.testing.assert_array_equal(value, np.arange(2))
            value = client.return_nothing(torch.tensor(10))

            # For now, "None" responses are empty tuples.
            self.assertEqual(value, ())
            self.assertEqual(client.nothing(), ())

        finally:
            server.stop()

    def test_bind_port_zero(self):
        server = postman.Server("127.0.0.1:0")
        server.run()
        try:
            # ephemeral port should be assigned
            self.assertNotEqual(server.port(), 0)
        finally:
            server.stop()

    def test_bind_unix_domain_socket(self):
        server = postman.Server("unix:/tmp/test.sock")
        server.run()
        try:
            self.assertNotEqual(server.port(), 0)
        finally:
            server.stop()

    def test_set_batch_size(self):
        address = "127.0.0.1"

        init_batch_size = 3
        final_batch_size = 2

        def run_client(port):
            client = postman.Client("%s:%i" % (address, port))
            client.connect(10)
            client.foo(torch.Tensor(init_batch_size, 2, 2))
            client.foo(torch.Tensor(final_batch_size, 2, 2))

        try:
            server = postman.Server("%s:0" % address)
            q = postman.ComputationQueue(batch_size=init_batch_size)
            server.bind_queue_batched("foo", q)
            server.run()

            client_proc = mp.Process(target=run_client, args=(server.port(),))
            client_proc.start()

            with q.get(wait_till_full=True) as batch:
                batch.set_outputs(batch.get_inputs()[0])

            q.set_batch_size(final_batch_size)

            with q.get(wait_till_full=True) as batch:
                batch.set_outputs(batch.get_inputs()[0])
        finally:
            q.close()
            server.stop()
            client_proc.join()


if __name__ == "__main__":
    unittest.main()
