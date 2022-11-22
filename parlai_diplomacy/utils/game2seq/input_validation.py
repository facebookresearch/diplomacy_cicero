#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import re
from typing import List, Dict

from fairdiplomacy.models.consts import POWERS
from parlai_diplomacy.utils.game2seq.format_helpers.misc import POT_TYPE_CONVERSION
from parlai_diplomacy.utils.game2seq.format_helpers.opt_utils import (
    expects_bilateral_pseudo_orders,
)
from parlai_diplomacy.utils.game2seq.typing import (
    DiplomacySequencePart,
    OrderSequencePart,
    DialogueSequencePart,
)


def _comment(regex: str, comment: str) -> str:
    return f"\n# {comment}\n" + regex + "\n"


class InputValidator:
    def __init__(
        self, format_parts: List[DiplomacySequencePart], output_type: str, opt: Dict, version: int
    ):
        if version > 3:
            raise NotImplementedError()

        self.format_parts = format_parts
        self.output_type = output_type
        self.opt = opt
        self.version = version
        self._set_vars()

    def _set_vars(self):
        self.ANY = r"([\s\S]*)"  # allows newlines

        self._POWER_NAMES = (
            [p.capitalize() for p in POWERS] if self.version <= 1 else POWERS + ["ALL"]
        )  # v2 allows ALL powers
        self.POWER = r"(" + r"|".join(self._POWER_NAMES) + r")"

        self.LOC_NOCOAST = r"([A-Z]{3})"
        self.LOC = r"([A-Z]{3}(/[NESW]C)?)"
        self.UNIT_NOCOAST = rf"(\*?[AF][ ]{self.LOC_NOCOAST})"
        self.UNIT = rf"(\*?[AF][ ]{self.LOC})"
        self.PHASE = r"([SFW]19[0-9][0-9][MRA])"
        self.M_PHASE = r"([SF]19[0-9][0-9]M)"
        self.RA_PHASE = r"([SFW]19[0-9][0-9][RA])"

        if self.version <= 1:
            self._ORDER_M = rf"{self.UNIT}[ ]-[ ]{self.LOC}([ ]VIA)?"
            self._ORDER_SMC = rf"{self.UNIT}[ ][CS][ ]{self.UNIT}[ ]-[ ]{self.LOC}"
        else:
            # V2 removes the '-'
            self._ORDER_M = rf"{self.UNIT}[ ]{self.LOC}([ ]VIA)?"
            self._ORDER_SMC = rf"{self.UNIT}[ ][CS][ ]{self.UNIT}[ ]{self.LOC}"
        self._ORDER_HBD = rf"{self.UNIT}[ ][HBD]"
        self._ORDER_R = rf"{self.UNIT}[ ]R[ ]{self.LOC}"
        # Our dataset has the property that support HOLDs always lack coastal qualifiers
        # So enforce that in our input validation to avoid surprise inconsistencies
        # in the formats that the model expects if we switch datasets
        self._ORDER_SH = rf"{self.UNIT}[ ]S[ ]{self.UNIT_NOCOAST}"
        self.ORDER = (
            r"("
            + rf"|".join(
                rf"({x})"
                for x in [
                    self._ORDER_HBD,
                    self._ORDER_M,
                    self._ORDER_R,
                    self._ORDER_SH,
                    self._ORDER_SMC,
                ]
            )
            + r")"
        )

        self.EO_O = rf"([ ]\[EO_O\])" if self.version <= 1 else ""
        self.EO_M = rf"([ ]\[EO_M\])" if self.version <= 1 else ""
        self.SEP = r"[ ]" if self.version <= 1 else r"[\n]"

        if self.version <= 1:
            self.ACTION = rf"(({self.ORDER}(;[ ]{self.ORDER})*)?{self.EO_O})"
        else:
            self.ACTION = rf"((({self.ORDER}(;[ ]{self.ORDER})*)?)?)"
        self.POWER_ACTION = rf"({self.POWER}:[ ]{self.ACTION})"
        self.BILATERAL_ACTION = _comment(
            rf"({self.POWER_ACTION}[\n]{self.POWER_ACTION})", "BILATERAL_ACTION"
        )

        if self.version <= 1:
            self.JOINT_ACTION = _comment(
                r"(" + r"[\n]".join([self.POWER_ACTION] * 7) + r")", "JOINT_ACTION"
            )
        else:
            # In V2, we do not necessarily include all powers in the joint action
            self.JOINT_ACTION = _comment(
                rf"({self.POWER_ACTION}(([\n]{self.POWER_ACTION})+)?)", "JOINT_ACTION"
            )

        if self.version <= 1:
            self.PY_LOC_LIST = rf"(\[('{self.LOC}'(,[ ]'{self.LOC}')*)?\])"
            self.PY_UNIT_RETREATS = rf"('{self.UNIT}':[ ]{self.PY_LOC_LIST})"
            self.PY_POWER_RETREATS_DICT = (
                rf"({{({self.PY_UNIT_RETREATS}(,[ ]{self.PY_UNIT_RETREATS})*)?}})"
            )
            self.POWER_RETREATS = rf"({self.POWER}:[ ]{self.PY_POWER_RETREATS_DICT})"
            self.RETREATS = _comment(
                rf"(retreats:[ ]{self.POWER_RETREATS}(;[ ]{self.POWER_RETREATS})(;[ ]{self.POWER_RETREATS})(;[ ]{self.POWER_RETREATS})(;[ ]{self.POWER_RETREATS})(;[ ]{self.POWER_RETREATS})(;[ ]{self.POWER_RETREATS}))",
                "RETREATS",
            )
        else:
            self.PY_LOC_LIST = rf"(({self.LOC}([ ]/[ ]{self.LOC})*)?)"
            self.PY_UNIT_RETREATS = rf"({self.UNIT}[ ]-[ ]{self.PY_LOC_LIST})"
            self.PY_POWER_RETREATS_DICT = (
                rf"(({self.PY_UNIT_RETREATS}(,[ ]{self.PY_UNIT_RETREATS})*)?)"
            )
            self.POWER_RETREATS = rf"({self.POWER}:[ ]{self.PY_POWER_RETREATS_DICT})"
            self.RETREATS = _comment(
                rf"(retreats:[ ]{self.POWER_RETREATS}((;[ ]{self.POWER_RETREATS})+)?)", "RETREATS",
            )

        self.ORDER_HISTORY_SINCE_LAST_MOVEMENT = rf"(({self.M_PHASE}[\n]{self.JOINT_ACTION}([\n]{self.RA_PHASE}[\n]({self.JOINT_ACTION})?)*)?)"  # N.B. "?" suffix, can be omitted in S1901M (but still invokes `sep`)

        self.ROLLOUT_ACTION = rf"({self.ACTION}(([\n]{self.RA_PHASE}[\n]{self.ACTION})?[\n]{self.M_PHASE}[\n]{self.ACTION})?)"  # 0-2 R/A phase actions + 1 mandatory M-phase action
        self.ROLLOUT_ALL_ACTION = (
            rf"({self.ACTION}"
            rf"([\n]{self.RA_PHASE}[\n]{self.ACTION})?"
            rf"([\n]{self.RA_PHASE}[\n]{self.ACTION})?"
            rf"([\n]{self.M_PHASE}[\n]{self.ACTION})?)"
        )  # 1 M-phase + 2 optional R/A phase actions + 1 optional M phase action
        # the final M phase is optional because the game might complete before we get there

        self.POWER_ROLLOUT_ACTION = _comment(
            rf"({self.POWER}:[ ]{self.ROLLOUT_ACTION})", "POWER_ROLLOUT_ACTION",
        )
        self.POWER_ROLLOUT_ALL_ACTION = _comment(
            rf"({self.POWER}:[ ]{self.ROLLOUT_ALL_ACTION})", "POWER_ROLLOUT_ALL_ACTION",
        )

        self.ROLLOUT_BILATERAL_ACTION = _comment(
            rf"({self.POWER_ROLLOUT_ACTION}[\n]{self.POWER_ROLLOUT_ACTION})",
            "ROLLOUT_BILATERAL_ACTION",
        )
        self.ROLLOUT_BILATERAL_ALL_ACTION = _comment(
            rf"({self.POWER_ROLLOUT_ALL_ACTION}[\n]{self.POWER_ROLLOUT_ALL_ACTION})",
            "ROLLOUT_BILATERAL_ALL_ACTION",
        )

        self.POWER_LOCS = rf"({self.POWER}:[ ]({self.LOC}(,[ ]{self.LOC})*)?)"
        self.POWER_UNITS = rf"({self.POWER}:[ ]({self.UNIT}(,[ ]{self.UNIT})*)?)"
        self.UNITS = rf"(units:[ ]{self.POWER_UNITS}(;[ ]{self.POWER_UNITS})*)"
        self.CENTERS = rf"(centers:[ ]{self.POWER_LOCS}(;[ ]{self.POWER_LOCS})*)"
        self.BUILDS = rf"(builds:([ ]{self.POWER}[ ]-?[0-9])+)"

        if self.version <= 1:
            self.LONGSTATE = rf"({self.ANY}\[EO_STATE\])"
            self.SHORTSTATE = rf"({self.UNITS}([\n]{self.RETREATS})?[ ]\[EO_STATE\])"
        else:
            self.LONGSTATE = self.ANY
            self.SHORTSTATE = rf"({self.UNITS}([\n]{self.RETREATS})?)"
            if self.opt.get("include_centers_state"):
                self.SHORTSTATE += rf"[\n]{self.CENTERS}"
            if self.opt.get("include_builds_state"):
                self.SHORTSTATE += rf"([\n]{self.BUILDS})?"  # optional, only in A-phase

        self.SLEEP_TIME = rf"([0-9]{{1,6}})"  # 6 digits allows 24 hours, but not unix timestamps.
        self.PLAYER_RATING = rf"([1-5])"
        self.FOR_POWER = rf"(for[ ]{self.POWER})"
        self.TO_POWER = rf"(->[ ]{self.POWER})"
        self.ANON = rf"((NON-)?ANON)"
        self.MINUTES = rf"([0-9]+min)"
        self.POT_TYPE = r"(" + r"|".join([rf"({x})" for x in POT_TYPE_CONVERSION.values()]) + r")"
        self.ALL_UNK = rf"(ALL-UNK)"
        self.DRAW_TYPE = rf"((PUBLIC)|(PRIVATE))"
        self.HAS_DRAWS = rf"((HASDRAWS)|(NODRAWS))"
        self.BAD = rf"(BAD)"

        self.MESSAGE = rf"({self.POWER}[ ]->[ ]{self.POWER}:[ ][^\n]*?{self.EO_M})"
        if self.version <= 1:
            self.CORRUPTION_CANDIDATE_MESSAGE = (
                rf"([\n]{self.PHASE}[\n]{self.POWER}[ ]->[ ]{self.POWER}:[ ][^\n]*?{self.EO_M})"
            )
        else:
            self.CORRUPTION_CANDIDATE_MESSAGE = rf"([\n]{self.POWER}:[ ][^\n]*?{self.EO_M})"
        self.SLEEP_MESSAGE = rf"({self.POWER}[ ]->[ ](Sleep):[ ]([0-9>]+){self.EO_M})"
        self.MESSAGE = (
            rf"({self.MESSAGE}|{self.SLEEP_MESSAGE})"
            if self.opt.get("include_sleep_messages", False)
            else self.MESSAGE
        )
        self.MESSAGE = (
            rf"({self.SLEEP_TIME}[ ]{self.MESSAGE})"
            if self.opt.get("add_sleep_times", False)
            else self.MESSAGE
        )

        if self.opt.get("hide_empty_draw_state"):
            self.DRAW_STATE = rf"([\n]DRAWS:[ ]({self.POWER}([ ]{self.POWER})*)?)?"
        else:
            self.DRAW_STATE = rf"(([\n]DRAWS:[ ]Unavailable)|([\n]DRAWS:[ ])|([\n]DRAWS:[ ]({self.POWER}([ ]{self.POWER})*)))"
        self.PHASE_MESSAGES = rf"({self.PHASE}([\n]{self.MESSAGE})+)"
        self.MESSAGE_HISTORY = rf"(({self.PHASE_MESSAGES}([\n]{self.PHASE_MESSAGES})*)?)"
        if self.opt.get("include_draw_state"):
            self.MESSAGE_HISTORY += _comment(self.DRAW_STATE, "DRAW_STATE")

        self.PHASE_MESSAGE = rf"([\n]{self.PHASE}[\n]{self.MESSAGE})"  # N.B. newline prefix

        # Set pseudo orders regex
        if self.opt.get("rollout_pseudo_orders") and not self.opt.get(
            "rollout_except_movement", True
        ):
            self.PSEUDO_ORDERS_REGEX = self.ROLLOUT_BILATERAL_ALL_ACTION
        elif self.opt.get("rollout_pseudo_orders"):
            self.PSEUDO_ORDERS_REGEX = self.ROLLOUT_BILATERAL_ACTION
        elif expects_bilateral_pseudo_orders(self.opt):
            self.PSEUDO_ORDERS_REGEX = self.BILATERAL_ACTION
        else:
            self.PSEUDO_ORDERS_REGEX = self.JOINT_ACTION

        # set actual orders regex
        self.ACTUAL_ORDERS = (
            self.ACTION
            if not self.opt.get("rollout_actual_orders")
            else self.ROLLOUT_ALL_ACTION
            if not self.opt.get("rollout_except_movement", True)
            else self.ROLLOUT_ACTION
        )

        self.STATE = self.SHORTSTATE if self.opt.get("shortstate", False) else self.LONGSTATE
        self.STATE_HISTORY_SINCE_LAST_MOVEMENT = rf"(({self.M_PHASE}[\n]{self.STATE}([\n]{self.RA_PHASE}[\n]({self.STATE})?)*)?)"  # N.B. "?" suffix, can be omitted in S1901M (but still invokes `sep`)

        # These parts may be omitted entirely -- e.g. if there is no message
        # history, the entire part is gone, including the separators
        self.OPTIONAL_BODY_PARTS = (
            OrderSequencePart.MESSAGE_HISTORY,
            DialogueSequencePart.HISTORY,
            DialogueSequencePart.ORDER_HISTORY_SINCE_LAST_MOVE,
        )

    # This maps format string parts to regex strings. If you add a new format
    # string part, you should add a regex validator here!
    def part_to_regex(self, part: DiplomacySequencePart) -> str:
        supported_mappings: Dict[DiplomacySequencePart, str] = {
            OrderSequencePart.STATE: self.STATE,
            OrderSequencePart.MESSAGE_HISTORY: self.MESSAGE_HISTORY,
            OrderSequencePart.ORDER_HISTORY_SINCE_LAST_MOVEMENT: self.ORDER_HISTORY_SINCE_LAST_MOVEMENT,
            DialogueSequencePart.HISTORY: self.MESSAGE_HISTORY,
            # If we want to condition on n movement phases, but we have only played through m < n movement phases so far, we shouln't expect duplicate separators.
            DialogueSequencePart.STATE: f"{self.SEP}?".join(
                [
                    self.STATE_HISTORY_SINCE_LAST_MOVEMENT
                    for _ in range(self.opt["extend_state_history_since_last_n_movement_phase"])
                ]
            )
            if self.opt.get("extend_state_history_since_last_n_movement_phase", 0) > 0
            else self.STATE,
            DialogueSequencePart.ORDER_HISTORY_SINCE_LAST_MOVE: f"{self.SEP}?".join(
                [
                    self.ORDER_HISTORY_SINCE_LAST_MOVEMENT
                    for _ in range(
                        self.opt.get("extend_order_history_since_last_n_movement_phase", 1)
                    )
                ]
            ),
            DialogueSequencePart.PSEUDO_ORDERS: self.PSEUDO_ORDERS_REGEX,
            DialogueSequencePart.ACTUAL_ORDERS: self.ACTUAL_ORDERS,
        }

        if part not in supported_mappings.keys():
            raise NotImplementedError(
                f"There is no regex validator written for {part}. "
                f"If this is a newly added part, a regex validator must be written for it (contact jgray for help)."
            )

        return supported_mappings[part]

    def get_input_validation_regex(self) -> str:
        class RegexBuilder:
            def __init__(self, sep: str):
                self.set_sep(sep)
                self.regex = ""

            def set_sep(self, sep: str):
                self.sep = sep

            def build(self) -> str:
                return self.regex

            def add(
                self,
                pattern: str,
                comment: str = "",
                optional: bool = False,
                omit_sep: bool = False,
            ):
                # prefix
                if optional or comment:
                    self.regex += "\n" + r"("
                    if comment:
                        self.regex += " # " + comment
                    self.regex += "\n"

                # add separator (can also always match start of string to account for optionals)
                if self.regex != "" and not omit_sep:
                    self.regex += rf"(^|{self.sep})"
                    self.regex += "\n"

                # add pattern
                self.regex += pattern

                # suffix
                if optional or comment:
                    self.regex += "\n" + r")"
                    if optional:
                        self.regex += r"? # OPTIONAL, INCLUDING SEP"
                    self.regex += "\n"

        regex = RegexBuilder(sep=self.SEP)

        # build the "body" (before the prompt) which is separated by `sep`
        for part in self.format_parts:
            regex.add(
                self.part_to_regex(part),
                comment=str(part),
                optional=(part in self.OPTIONAL_BODY_PARTS),
            )

        # build the prompt. Set the sep to " " after the first token
        if self.opt.get("add_sleep_times"):
            regex.add(self.SLEEP_TIME, comment="SLEEP_TIME")
        regex.add(self.PHASE, comment="PHASE")
        regex.set_sep("[ ]")
        regex.add(self.POWER, comment="POWER")
        if "plausiblepseudoorder" in self.output_type or self.opt.get(
            "add_recipient_to_prompt", False
        ):
            regex.add(self.TO_POWER, comment="TO_POWER")
        if self.opt.get("include_player_ratings"):
            regex.add(self.PLAYER_RATING, comment="PLAYER_RATING")
        if self.opt.get("include_game_info"):
            regex.add(self.ANON, comment="ANON")
            regex.add(self.MINUTES, comment="MINUTES")
            regex.add(self.POT_TYPE, comment="POT_TYPE")
            regex.add(self.ALL_UNK, optional=True, comment="MAYBE ALL-UNK")
        if self.opt.get("include_draw_info"):
            regex.add(self.DRAW_TYPE, comment="DRAW_TYPE")
            regex.add(self.HAS_DRAWS, comment="HAS_DRAWS")
        if "independent" in self.output_type or self.output_type == "sleepsix":
            regex.add(self.FOR_POWER, comment="FOR_POWER")

        regex.add(":", comment="PROMPT", omit_sep=True)

        if self.output_type == "humanvsmodeldiscriminator":
            regex.add(self.ANY, comment="POSSIBLY BADLY FORMATTED MESSAGE", omit_sep=True)
        elif "dialoguediscriminator" in self.opt.get("task", ""):
            regex.add(
                self.CORRUPTION_CANDIDATE_MESSAGE, comment="CANDIDATE MESSAGE", omit_sep=True
            )

        # some parts go after the colon
        if self.opt.get("mark_bad_messages"):
            regex.set_sep("[ ]")
            regex.add(self.BAD, optional=True, comment="MAYBE BAD")

        return _comment(regex.build(), "TASK: " + self.opt.get("task", ""))


class InputValidationException(ValueError):
    pass


def validate(regex: str, x: str, throw_on_failure: bool = True) -> None:
    if not re.fullmatch(regex, x, re.VERBOSE):
        msg = f"\n\nREGEX:\n{regex}\n\nINPUT:\n{x}\n\nERROR: InputValidationException"
        if throw_on_failure:
            raise InputValidationException(msg)
        else:
            logging.error(msg)
