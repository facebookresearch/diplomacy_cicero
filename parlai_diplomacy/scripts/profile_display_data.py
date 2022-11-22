#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Run the python profiler on display data and prints the results.
```
python parlai_diplomacy/scripts/profile_display_data.py --truncate -1 -t convai2
```
"""
import cProfile
import io
import pstats
import re

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.scripts.display_data import setup_args as display_data_args
from parlai.scripts.display_data import display_data
import parlai.utils.logging as logging

import parlai_diplomacy.utils.loading as load


load.register_all_agents()
load.register_all_tasks()


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, "cProfile display data")
    parser = display_data_args(parser)
    profile = parser.add_argument_group("Profiler Arguments")
    profile.add_argument(
        "--debug", type="bool", default=False, help="If true, enter debugger at end of run.",
    )
    profile.add_argument(
        "--truncate", type=int, default=50, help="Truncate output lines. Use -1 for no truncation",
    )
    profile.add_argument(
        "--subtract-iterdirs",
        type="bool",
        default=False,
        help="If true, subtract the time it takes to iterate through directories",
    )
    parser.set_defaults(
        task="message_history_shortstate_pseudoorder_dialogue_chunk",
        single_view_pseudo_orders=True,
        include_game_info=True,
        add_sleep_times=True,
    )
    return parser


def profile(opt: Opt) -> float:
    """
    Profile display data

    Returns the total cumulative time.
    """
    # Profile display data
    pr = cProfile.Profile()
    pr.enable()
    display_data(opt)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")

    # Log info
    ps.print_stats()
    lines = s.getvalue().split("\n")

    # Subtract the cumulative time from the iterdirs call from the overall
    # cumulative time. We do this because the time it takes to do this can
    # be variable, depending on filesystem caching.
    if opt["subtract_iterdirs"]:
        try:
            iterdir_time = sum(
                [float(re.split(" +", line.strip())[3]) for line in lines if "iterdir" in line]
            )  # cum time for iterdirs call
        except IndexError:
            print([line for line in lines if "iterdir" in lines])
            raise
    else:
        iterdir_time = 0.0

    first_line = lines[0]
    if opt["truncate"] > 0:
        lines = lines[: opt["truncate"]]

    logging.success(f"Finished profiling for task: {opt['task']}")
    logging.info("\n".join(lines))
    if opt["truncate"] > 0:
        logging.warning(f"...Output truncated to {opt['truncate']} lines")
    logging.success(first_line)

    # Return the total cumulative time
    call_seconds = float(first_line.split(" ")[-2])
    call_seconds = call_seconds - iterdir_time
    if iterdir_time > 0:
        logging.success(f"\tExcluding iterdirs time: {call_seconds}s")

    return call_seconds


class ProfileDisplayData(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return profile(self.opt)


if __name__ == "__main__":
    ProfileDisplayData.main()
