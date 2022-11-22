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


# A server example showing some more postman stuff.


def pyfunc(t):
    return 42 * (t + 1)


def identity(arg):
    print(arg)
    return arg


def main():
    # TODO: Re-add TorchScript modules. Example code:
    # https://github.com/fairinternal/torchbeast/blob/4e34d2b6493ea2f2d364e8cd7c5eb9596b9dcb6d/torchbeast/server.cc#L185

    # module = torch.jit.script(torch.nn.Linear(2, 3))

    server = postman.Server("localhost:12345")

    # s.bind("mymodule", module)
    server.bind("pyfunc", pyfunc, batch_size=1)
    server.bind("identity", identity, batch_size=1)
    server.bind("batched_identity", identity, batch_size=2, wait_till_full=True)

    # server.bind("batched_myfunc", module, batch_size=2)

    # Alternative: Binding "ComputationQueue"s instead of functions directly:
    queue = postman.ComputationQueue(batch_size=2)
    server.bind_queue("batched_identity2", queue)

    def read_queue():
        try:
            while True:
                with queue.get(wait_till_full=False) as batch:
                    batch.set_outputs(identity(*batch.get_inputs()))
        except StopIteration:
            return

    thread = threading.Thread(target=read_queue)
    thread.start()

    server.run()

    try:
        while True:
            time.sleep(1)  # Could also deal with signals. I guess.
    except KeyboardInterrupt:
        queue.close()
        server.stop()
        server.wait()
        thread.join()


if __name__ == "__main__":
    main()
