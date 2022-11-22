#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""A small utility for sending slack messages to the Diplomacy workspace.

To enable sending messages to an external service:
    1. Grab webhook URL(s) for whatever external service you want to post to.
    2. Create local directory to store webhooks and assign SLACK_SECRETS_DIR to its path.
    3. Create text files in this directory with each having a single webhook URL,
        and the title of each file being a chosen "nickname" for that file's respective URL.
    4. When hitting send_slack_message below set the "channel" arg to the filename / "nickname"
        of the respective webhook you want to target.
"""
import io
import logging
import os
import platform
import traceback
from glob import glob
from typing import Optional

import requests

SLACK_SECRETS_DIR = None
INCOMING_WEBHOOKS = {}

if SLACK_SECRETS_DIR is not None:
    for g in glob(SLACK_SECRETS_DIR + "/*"):
        INCOMING_WEBHOOKS[g.split("/")[-1]] = open(g).read()


def send_slack_message(channel: str, message: str) -> bool:
    """Returns True on success. Catch and log all errors."""
    try:
        if channel not in INCOMING_WEBHOOKS:
            logging.error(
                f"Cannot send message to {channel}. Available channels: {list(INCOMING_WEBHOOKS.keys())}"
            )
            return False
        r = requests.post(INCOMING_WEBHOOKS[channel], json={"text": message})
        if not r.ok:
            logging.error(f"send_slack_message to {channel} failed: {r.status_code} {r.content}")
        return r.ok
    except Exception as e:
        logging.exception(f"send_slack_message to {channel} failed:")
        return False


class SlackExceptionSwallower:
    """Context manager which swallows exceptions and logs them to slack

    By default, a new object is deactivated, and exceptions will not be swallowed.

    After activate() is called, exceptions will be logged to slack and swallowed.

    In the fairdiplomacy repo, this is used by importing the GLOBAL_SLACK_EXCEPTION_SWALLOWER
    below and activating it once in entrypoints (grep for example(s)).
    """

    def __init__(self):
        self._slack_channel: Optional[str] = None
        self._debug_str: Optional[str] = None

    def activate(self, slack_channel: str = "exceptions", debug_str: Optional[str] = None):
        self._slack_channel = slack_channel
        self._debug_str = debug_str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self._slack_channel is not None and exc_type is not None:
            logging.error("SlackExceptionSwallower caught: %s", exc, exc_info=True)
            with io.StringIO() as s:
                print(
                    f"pid={os.getpid()} host={platform.node()} slurm_jobid={os.environ.get('SLURM_JOB_ID', None)}",
                    file=s,
                )
                if self._debug_str:
                    print(self._debug_str, file=s)
                print("```", file=s)
                traceback.print_exception(exc_type, exc, exc_tb, file=s)
                print("```", file=s)
                send_slack_message(self._slack_channel, s.getvalue())
                return True  # swallow exception


GLOBAL_SLACK_EXCEPTION_SWALLOWER = SlackExceptionSwallower()
