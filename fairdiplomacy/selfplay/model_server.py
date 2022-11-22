#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import time
from typing import Optional

import fairdiplomacy.selfplay.ckpt_syncer
import numpy as np
import postman
import torch

from fairdiplomacy.utils.multiprocessing_spawn_context import get_multiprocessing_ctx
from fairdiplomacy.utils.timing_ctx import TimingCtx

mp = get_multiprocessing_ctx()


def run_server(port, batch_size, port_q=None, **kwargs):
    def set_seed(seed):
        seed = seed.item()
        logging.info(f"Set server seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)
    max_port = port + 10
    try:
        logging.info(f"Starting server port={port} batch={batch_size}")
        eval_queue = postman.ComputationQueue(batch_size)
        for p in range(port, max_port):
            server = postman.Server(f"127.0.0.1:{p}")
            server.bind(
                "set_batch_size", lambda x: eval_queue.set_batch_size(x.item()), batch_size=1
            )
            server.bind("set_seed", set_seed, batch_size=1)
            server.bind_queue_batched("evaluate", eval_queue)
            try:
                server.run()
                break  # port is good
            except RuntimeError:
                continue  # try a different port
        else:
            raise RuntimeError(f"Couldn't start server on ports {port}:{max_port}")

        bound_port = server.port()
        assert bound_port != 0

        logging.info(f"Started server on port={bound_port} pid={os.getpid()}")
        if port_q is not None:
            port_q.put(bound_port)  # send port to parent proc

        server_handler(eval_queue, **kwargs)
    except Exception as e:
        logging.exception("Caught exception in the server (%s)", e)
        raise
    finally:
        eval_queue.close()
        server.stop()


def server_handler(
    q: postman.ComputationQueue,
    load_model_fn,
    seed,
    output_transform=None,
    device: Optional[int] = 0,
    ckpt_sync_path=None,
    ckpt_sync_every=0,
    wait_till_full=False,
    empty_cache=True,
    input_transform=None,
    model_path=None,  # for debugging only
):

    if not torch.cuda.is_available() and device is not None:
        logging.warning(f"Cannot run on GPU {device} as not GPUs. Will do CPU")
        device = None
    if device is not None and device > 0:
        # device=[None] stands for CPU.
        torch.cuda.set_device(device)
    model = load_model_fn()
    logging.info(f"Server {os.getpid()} loaded model, device={device}, seed={seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    frame_count, batch_count, total_batches = 0, 0, 0
    timings = TimingCtx()
    totaltic = time.time()

    if ckpt_sync_path is not None:
        ckpt_syncer = fairdiplomacy.selfplay.ckpt_syncer.CkptSyncer(ckpt_sync_path)
        last_ckpt_version = ckpt_syncer.maybe_load_state_dict(model, last_version=None)
        if ckpt_sync_every:
            next_ckpt_sync_time = time.time() + ckpt_sync_every

    with torch.no_grad():
        while True:
            if empty_cache and device is not None:
                torch.cuda.empty_cache()
            try:
                with q.get(wait_till_full=wait_till_full) as batch:
                    with timings("ckpt_sync"):
                        if ckpt_sync_path is not None and (
                            not ckpt_sync_every or time.time() >= next_ckpt_sync_time
                        ):
                            last_ckpt_version = ckpt_syncer.maybe_load_state_dict(
                                model, last_ckpt_version
                            )
                            if ckpt_sync_every:
                                next_ckpt_sync_time = time.time() + ckpt_sync_every

                    with timings("next_batch"):
                        inputs = batch.get_inputs()[0]
                        if input_transform is not None:
                            inputs = input_transform(**inputs)

                    with timings("to_cuda"):
                        if device is not None:
                            inputs = {
                                k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()
                            }

                    with timings("model"):
                        y = model(**inputs)

                    with timings("transform"):
                        if output_transform is not None:
                            y = output_transform(inputs, y, model)

                    with timings("to_cpu"):
                        y = tuple(x.to("cpu") for x in y)

                    with timings("reply"):
                        batch.set_outputs(y)

                    # Do some performance logging here
                    batch_count += 1
                    total_batches += 1
                    frame_count += inputs["x_board_state"].shape[0]
                    if total_batches > 16 and (total_batches & (total_batches - 1)) == 0:
                        delta = time.time() - totaltic
                        logging.info(
                            f"Server thread: performed {batch_count} forwards of avg batch size {frame_count / batch_count} "
                            f"in {delta} s, {frame_count / delta} forward/s."
                        )
                        TimingCtx.pprint_multi([timings], logging.info)
                        batch_count = frame_count = 0
                        timings.clear()
                        totaltic = time.time()

            except TimeoutError as e:
                logging.info("TimeoutError: %s", e)

    logging.info("SERVER DONE")
