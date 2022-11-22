#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple, Dict

from parlai.core.loader import register_teacher

from fairdiplomacy.typedefs import Phase, Power
from parlai_diplomacy.metrics.classifiers import ClassifierMetricMixin
from parlai_diplomacy.tasks.base_diplomacy_agent import BaseDiplomacyTeacher
from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.utils.game2seq.format_helpers.misc import modify_input_prompt_for_power


@lru_cache(3000)
def _get_labels_for_time(classes: Tuple[str], time: str, remaining_time: float):
    if remaining_time < 0:
        # Not sure what causes this
        return None
    # Map a time like "1234" or ">50" to a set of labels.
    if time[0] == ">":
        # If the time is ">n", any time greater than n is a valid label, but we
        # also exclude times that are more than the remaining time in the phase
        lower_bound = float(time[1:])
        return [
            t
            for t in classes[1:]  # self.classes[0] is pad
            if float(t) > lower_bound and (t == "inf" or float(t) < remaining_time)
        ]
    elif time == "inf" or int(time) <= remaining_time:
        # This just ignores sleep times that are outside our space of labels.
        # We could also reasonably make these by 'inf', or use BPE in the
        # labels to let us represent any length of time.
        for t in classes[1:]:
            if float(t) >= float(time):
                return [t]
    else:
        return None


@register_teacher("message_history_sleepclassifier_chunk")
@register_teacher("message_history_shortstate_sleepclassifier_chunk")
@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_sleepclassifier_chunk"
)
class SleepClassifierChunkTeacher(ClassifierMetricMixin, BaseDialogueChunkTeacher):
    """
    Streaming data base Sleep-Classifier teacher.

    Input: message info + game-related info (state, order, etc)
    Output: time to sleep for
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser = BaseDiplomacyTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)
        argparser.set_defaults(
            include_sleep_messages=True,
            add_sleep_times=True,
            # Hard codinga list of roughly equally frequent bins of ties (in seconds)
            classes="pad 0 15 35 64 113 207 385 698 1220 2060 3340 5300 8390 13500 22600 39400 86000 inf".split(
                " "
            ),
            include_game_info=True,
        )

        return argparser

    def __init__(self, opt, shared=None):
        if not opt.get("include_sleep_messages", None):
            raise RuntimeError("Must include sleep messages")
        if not opt.get("include_game_info", None):
            raise RuntimeError("Must set --include-game-info True")
        super().__init__(opt, shared)
        self.id = "Base Sleep-Classifier Chunk"
        self.classes = tuple(opt["classes"])

    @property
    def model_type(self) -> str:
        return "sleep_classifier"

    def _generate_example_tuples(self, game, game_id):
        def make_time_string(toks: List[str]) -> str:
            assert toks[0] in ["inf", "IN", "OUT"], toks
            if toks[0] == "inf":
                return "inf"
            else:
                return (">" if toks[0] == "IN" else "") + toks[1]

        # Official phase time may be exceeded due to phases being extended,
        # trivial-R phase messages being shoved to the next M-phase, or
        # possibly other reasons. Here we ensure the phase time (used to
        # compute time remaining) at least covers the messages in that phase.
        official_phase_time_seconds = (
            float(self.get_player_metadata(game, game_id)["phase_minutes"]) * 60
        )
        message_history = game.message_history
        phase_time_lower_bounds_seconds: Dict[Phase, int] = {
            phase: (max(messages) - min(messages)).to_seconds_int()
            for phase, messages in message_history.items()
        }

        examples = super()._generate_example_tuples(game, game_id)
        result = []
        player_and_phase_to_elapsed_time = defaultdict(lambda: 0.0)
        for ex in examples:
            phase = ex["phase_id"]
            phase_seconds = max(
                official_phase_time_seconds, phase_time_lower_bounds_seconds.get(phase, 0)
            )
            lines = ex["labels"][0].split("\n")
            if self.opt["task_version"] == 1:
                # Example:
                # """
                # S1901M
                # England -> Sleep: 28 [EO_M]
                # """
                if len(lines) < 2 or " -> Sleep:" not in lines[1]:
                    continue
                toks = lines[1].strip().split()
                assert toks[2] == "Sleep:", toks
                time_string = make_time_string(toks[3:])
            else:
                # Example:
                # """
                # sleep: >10
                # """
                if len(lines) != 1 or not lines[0].startswith("sleep: "):
                    continue
                toks = lines[0].split("sleep: ")[1].split()
                time_string = make_time_string(toks)

            labels = _get_labels_for_time(
                self.classes,
                time_string,
                # The remaining time in the phase is the total time per phase,
                # minus the sum of this player's sleep times
                remaining_time=(
                    phase_seconds - player_and_phase_to_elapsed_time[ex["player"], ex["phase_id"]]
                ),
            )
            # For each player, track how much time has elapsed as the sum of their sleep times
            player_and_phase_to_elapsed_time[ex["player"], ex["phase_id"]] += float(
                time_string if time_string[0] != ">" else time_string[1:]
            )

            if labels is not None:
                ex["labels"] = labels
                result.append(ex)
        return result


@register_teacher("message_history_orderhistorysincelastmovementphase_shortstate_sleepsix_chunk")
class SleepSixChunkTeacher(ClassifierMetricMixin, BaseDialogueChunkTeacher):
    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser = BaseDiplomacyTeacher.add_cmdline_args(argparser, partial_opt=partial_opt)
        argparser.set_defaults(
            # Hard coding list of roughly equally frequent bins of ties (in seconds)
            classes=tuple(
                "pad 0 15 35 64 113 207 385 698 1220 2060 3340 5300 8390 13500 22600 39400 86000 inf".split()
            ),
            include_sleep_messages=True,
            task_version=3,
            add_sleep_times=True,
            hide_empty_draw_state=True,
            include_centers_state=True,
            include_draw_info=True,
            include_draw_state=True,
            include_game_info=True,
            include_player_ratings=True,
        )

        return argparser

    def __init__(self, opt, shared=None):
        if not opt.get("include_sleep_messages", None):
            raise RuntimeError("Must include sleep messages")
        if not opt.get("include_game_info", None):
            raise RuntimeError("Must set --include-game-info True")
        assert opt.get("task_version", 1) >= 2, opt.get("task_version")
        super().__init__(opt, shared)
        self.id = "Base Sleep-Classifier Chunk"
        self.classes = opt["classes"]

    @property
    def model_type(self) -> str:
        return "sleepsix"

    def _generate_example_tuples(self, game, game_id):
        phase_minutes = float(self.get_player_metadata(game, game_id)["phase_minutes"])

        alive_powers_by_phase = {
            p: game.rolled_back_to_phase_start(p).get_alive_powers()
            for p in game.get_all_phase_names()
        }

        examples = super()._generate_example_tuples(game, game_id)
        result = []
        player_and_phase_to_elapsed_time = defaultdict(lambda: 0.0)
        for ex in examples:
            label = ex["labels"][0]
            toks = label.split()
            if toks[0] != "sleep:":
                continue
            alive_powers = [
                p for p in alive_powers_by_phase[ex["phase_id"]] if p != ex["player"]
            ] + ["ALL"]
            if toks[-1] == "inf":
                # an inf sleep is for everybody
                time_strings_and_recipients = [("inf", p) for p in alive_powers]
            elif toks[1] == "IN":
                # on an inbound message, we have a lower bound for everybody
                time_strings_and_recipients = [(">" + toks[2], p) for p in alive_powers]
            elif toks[1] == "OUT":
                # on an outbound message, we have an exact time for the
                # recipient and a lower bound for everybody else
                recipient = toks[3]
                time_strings_and_recipients = [(">" + toks[2], p) for p in alive_powers]
                time_strings_and_recipients.append((toks[2], recipient))
            else:
                raise Exception(f"Bad sleep message: {label}")

            # The remaining time in the phase is the total time per phase,
            # minus the sum of this player's sleep times
            remaining_time = (phase_minutes * 60) - player_and_phase_to_elapsed_time[
                (ex["player"], ex["phase_id"])
            ]

            for time_string, recipient in time_strings_and_recipients:
                labels = _get_labels_for_time(
                    self.classes, time_string, remaining_time=remaining_time
                )
                if labels is not None:
                    _ex = copy.copy(ex)
                    _ex["labels"] = labels
                    assert _ex["text"].endswith(":"), _ex["text"]
                    # add recipient to prompt
                    _ex["text"] = modify_input_prompt_for_power(
                        _ex["text"], recipient, self.opt["task_version"]
                    )
                    result.append(_ex)

            # For each player, track how much time has elapsed as the sum of their sleep times
            time_string = time_string  # type:ignore variable is defined
            player_and_phase_to_elapsed_time[ex["player"], ex["phase_id"]] += float(
                time_string if time_string[0] != ">" else time_string[1:]
            )

        return result
