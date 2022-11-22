#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""A service that can accept metrics from other workers and send them to wandb/tb/jsonl.

The main process should initialize MetricLoggingServer that will spawn a
logging process. This process will start a ZMQ server on random port.

The clients should call get_remote_logger and log as usual. Under the
hood it will connect to the server (assuming it's running on the slurm master
machine) and forward all metrics there.
"""
from typing import Dict, Optional
import logging
import os
import pathlib
import time

import zmq

from conf import conf_cfgs
from fairdiplomacy.selfplay.metrics import recursive_tensor_item, Logger
from fairdiplomacy.selfplay.wandb import initialize_wandb_if_enabled
from fairdiplomacy.selfplay.paths import get_remote_logger_port_file
from fairdiplomacy.utils.exception_handling_process import ExceptionHandlingProcess
import heyhi


def get_remote_logger(*, is_dummy: bool = False, tag=None) -> "RemoteLogger":
    return RemoteLogger(tag, is_dummy=is_dummy)


class RemoteLogger:
    """A class that shares interface with Logger, but actually sends data to remote."""

    def __init__(self, tag: Optional[str], is_dummy: bool = False):
        self._tag = tag
        self._dummy = is_dummy
        # Socket will be initialized lazily to guarantee the server has time to start.
        self._socket = None
        self._context = None

    def _maybe_connect(self):
        port_file = get_remote_logger_port_file()
        if not self._dummy and self._socket is None:
            for i in range(5):
                if port_file.exists():
                    break
                logging.warning("Cannot find %s. Sleeping for %s seconds", port_file, 2 ** i)
                time.sleep(2 ** i)
            with port_file.open() as stream:
                logger_address = stream.read().strip()

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUSH)
            self._socket.connect(logger_address)

    def log_config(self, cfg):
        if not self._dummy:
            self._maybe_connect()
            self._socket.send_json(dict(type="config", cfg=str(cfg), tag=self._tag))

    def log_metrics(self, metrics, step, sanitize=False):
        del sanitize  # Ignored. Always sanitize.
        if not self._dummy:
            metrics = recursive_tensor_item(metrics)
            self._maybe_connect()
            self._socket.send_json(dict(type="metrics", metrics=metrics, step=step, tag=self._tag))

    def stop_remote(self):
        # Not a logging command. Used to ask server to stop.
        self._maybe_connect()
        self._socket.send_json(dict(type="stop", tag=self._tag))

    def close(self):
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        if self._context is not None:
            self._context.destroy()
        self._context = None

    def __del__(self):
        self.close()


class MetricLoggingServer:
    """A class that accepts metrics dicts as zmq and writes them to json/tb/wandb.

    Note, if wandb is used, it's up the caller to initilize WANDB_RUN_GROUP
    prior to launching the server.
    """

    LAUNCHED = False

    def __init__(self, cfg: conf_cfgs.ExploitTask, default_project_name: str):
        assert heyhi.is_master(), "Only the master should start the server"
        assert not MetricLoggingServer.LAUNCHED
        MetricLoggingServer.LAUNCHED = True
        self.p = ExceptionHandlingProcess(
            target=self.run_metric_zmq_server,
            kwargs=dict(cfg=cfg, default_project_name=default_project_name),
        )
        self.p.start()

    def terminate(self) -> None:
        if self.p is not None:
            get_remote_logger().stop_remote()
            time.sleep(0.1)
            self.p.kill()
            self.p = None

    @staticmethod
    def run_metric_zmq_server(cfg: conf_cfgs.ExploitTask, default_project_name: str) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = socket.bind_to_random_port("tcp://*")
        port_file = get_remote_logger_port_file()
        tmp_port_file = pathlib.Path(str(port_file) + ".tmp")
        with tmp_port_file.open("w") as stream:
            print("tcp://%s:%s" % (heyhi.get_slurm_master(), port), file=stream)
        os.rename(tmp_port_file, port_file)

        log_wandb = initialize_wandb_if_enabled(cfg, default_project_name)

        loggers_per_tag: Dict[Optional[str], Logger] = {}
        logging.info("Starting run_metric_zmq_server loop (port=%s)", port)
        try:
            while True:
                message = socket.recv_json()
                message_type = message.pop("type")
                tag = message.pop("tag")
                if tag not in loggers_per_tag:
                    loggers_per_tag[tag] = Logger(
                        tag=tag or None, is_master=True, log_wandb=log_wandb
                    )
                if message_type == "metrics":
                    loggers_per_tag[tag].log_metrics(**message)
                elif message_type == "config":
                    loggers_per_tag[tag].log_config(**message)
                elif message_type == "stop":
                    logging.warning("Got a stop message")
                    break
                else:
                    raise RuntimeError(f"Bad message_type: {message_type}")
        except Exception:
            logging.exception("Got an exception in MetricLoggingServer")
        finally:
            logging.info("Cleaning up")
            socket.close()
            context.destroy()
            for logger in loggers_per_tag.values():
                logger.close()
        logging.info("Stopping run_metric_zmq_server loop")
