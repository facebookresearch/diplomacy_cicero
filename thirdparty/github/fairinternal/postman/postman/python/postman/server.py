#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import threading

from postman import rpc


def _target(queue, function, wait_till_full):
    try:
        while True:
            with queue.get(wait_till_full=wait_till_full) as batch:
                batch.set_outputs(function(*batch.get_inputs()))
    except StopIteration:
        return


class Server(rpc.Server):
    def bind(self, name, function, batch_size, num_threads=1, wait_till_full=False):
        self.threads = getattr(self, "threads", [])
        self.queues = getattr(self, "queues", [])
        queue = rpc.ComputationQueue(batch_size)
        self.bind_queue(name, queue)
        self.queues.append(queue)

        for i in range(num_threads):
            self.threads.append(
                threading.Thread(
                    target=_target,
                    name="thread-%s-%i" % (name, i),
                    args=(queue, function, wait_till_full),
                )
            )

    def stop(self):
        for queue in getattr(self, "queues", []):
            queue.close()
        super(Server, self).stop()

    def run(self):
        super(Server, self).run()
        for thread in getattr(self, "threads", []):
            thread.start()

    def wait(self):
        super(Server, self).wait()
        for thread in getattr(self, "threads", []):
            thread.join()
