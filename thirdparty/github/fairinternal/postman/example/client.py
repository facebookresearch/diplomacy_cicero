#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import numpy as np
import torch

import postman


def main():
    client_id = random.randint(0, 10000)
    print("Client with random id", client_id)

    client = postman.Client("localhost:12345")
    client.connect(deadline_sec=3)

    output = client.pyfunc(torch.zeros(1, 2))

    client_array = torch.tensor([0, client_id, 2 * client_id])
    inputs = (torch.tensor(0), torch.tensor(1), (client_array, torch.tensor(True)))
    client.identity(inputs)

    # Test that we get back what we expect.
    np.testing.assert_array_equal(client_array, client.identity(client_array))
    np.testing.assert_array_equal(client_array, client.batched_identity(client_array))


if __name__ == "__main__":
    main()
