#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
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


def main():
    server = postman.Server("%s:%d" % ("localhost", 12345))
    model = Model()
    replay_buffer = buffer.NestPrioritizedReplay(1000, 0, 0.6, 0.4, True)

    model_queue = buffer.ModelQueue(model)

    def add_replay(content, priority):
        print(content)
        print(priority)
        replay_buffer.add_batch_async(content, priority[0])

    def query_state_dict():
        # print(agent.state_dict())
        model_id, cur_model = model_queue.get_model()
        model_nest = nest.map(lambda t: t.unsqueeze(0), cur_model.state_dict())
        model_queue.release_model(model_id)
        return model_nest

    server.bind("query_state_dict", query_state_dict, batch_size=1)
    server.bind("add_replay", add_replay, batch_size=1)
    server.run()

    try:
        while True:
            time.sleep(1)
            print("current replay buffer size is %d" % replay_buffer.size())
    except KeyboardInterrupt:
        server.stop()
        server.wait()


if __name__ == "__main__":
    main()
