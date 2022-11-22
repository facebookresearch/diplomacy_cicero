#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import threading

import torch
import postman


def pyfunc(t):
    return 42 * (t + 1)


def identity(arg):
    print(arg)
    return arg


def main():
    server = postman.Server("localhost:12345")

    server.bind("pyfunc", pyfunc, batch_size=1)
    server.bind("identity", identity, batch_size=1)
    server.bind("batched_identity", identity, batch_size=2, wait_till_full=True)

    server.run()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        server.wait()


if __name__ == "__main__":
    main()
