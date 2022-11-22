#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import numpy as np
import torch
import nest
import postman
import buffer


def main():
    client = postman.Client("%s:%d" % ("localhost", 12345))
    client.connect(deadline_sec=10)
    local_replay_buffer = buffer.NestPrioritizedReplay(1000, 0, 0.6, 0.4, True)

    data = {}
    data["a"] = torch.Tensor(10)

    # testing, this could be a long-running c++ replay buffer adding
    local_replay_buffer.add_one(data, 1)
    local_replay_buffer.add_one(data, 2)

    time.sleep(1)
    size, batch, priority = local_replay_buffer.get_new_content()
    print(batch)

    client.add_replay(batch, priority)
    model = client.query_state_dict()
    print(model)


if __name__ == "__main__":
    main()
