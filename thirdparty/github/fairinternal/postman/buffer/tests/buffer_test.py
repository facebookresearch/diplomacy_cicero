#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import unittest
import multiprocessing as mp
import time
import numpy as np

import torch
import nest
import postman
import buffer


class Model(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc(x)


class BufferTest(unittest.TestCase):
    def test_add_replay(self, num_clients=2, address="localhost:23456"):
        def run_client():
            client = postman.Client(address)
            client.connect(10)
            local_replay_buffer = buffer.NestPrioritizedReplay(1000, 0, 0.6, 0.4, True)

            data = {}
            data["a"] = torch.Tensor(10)

            # testing, this could be a long-running c++ replay buffer adding
            local_replay_buffer.add_one(data, 1)
            local_replay_buffer.add_one(data, 2)

            size, batch, priority = local_replay_buffer.get_new_content()
            client.add_replay(batch, priority)

        client_processes = [mp.Process(target=run_client) for _ in range(num_clients)]

        replay_buffer = buffer.NestPrioritizedReplay(1000, 0, 0.6, 0.4, True)

        def add_replay(content, priority):
            replay_buffer.add_batch_async(content, priority[0])

        server = postman.Server(address)
        server.bind("add_replay", add_replay, batch_size=1)
        server.run()

        for p in client_processes:
            p.start()

        for p in client_processes:
            p.join()

        server.stop()

        self.assertEqual(replay_buffer.size(), 2 * num_clients)

    def test_query_model(self, num_clients=2, address="localhost:23457"):
        def run_client():
            client = postman.Client(address)
            client.connect(10)
            model = client.query_state_dict()
            self.assertEqual(model["fc.weight"].size, 100)
            self.assertEqual(model["fc.bias"].size, 10)

        model = Model()
        model_queue = buffer.ModelQueue(model)

        def query_state_dict():
            # print(agent.state_dict())
            model_id, cur_model = model_queue.get_model()
            model_nest = nest.map(lambda t: t.unsqueeze(0), cur_model.state_dict())
            model_queue.release_model(model_id)
            return model_nest

        server = postman.Server(address)
        server.bind("query_state_dict", query_state_dict, batch_size=1)
        server.run()

        client_processes = [mp.Process(target=run_client) for _ in range(num_clients)]

        for p in client_processes:
            p.start()

        for p in client_processes:
            p.join()

        server.stop()


if __name__ == "__main__":
    unittest.main()
