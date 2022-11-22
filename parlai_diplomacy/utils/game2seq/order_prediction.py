#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Format helper for order prediction tasks.
"""
from abc import ABC, abstractmethod
import contextlib
from typing import List, Optional, Tuple, Dict, Sequence
import random

from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import Action, GameJson, JointAction, Phase, Power, RolloutJointAction

from parlai_diplomacy.utils.game2seq.base_prediction import BaseDiplomacyPredictionFormatter
from parlai_diplomacy.utils.game2seq.dialogue_prediction import (
    DialoguePredictionFormatter,
    TrainingDialoguePredictionFormatter,
)
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    MessageObjectPart,
    get_gamejson_draw_state,
    get_last_timestamp_msg_history,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    format_player_prompt_token,
    add_recipient_to_prompt_token,
    COUNTRY_ID_TO_POWER,
)
import parlai_diplomacy.utils.game2seq.format_helpers.orders as order_helper
from parlai_diplomacy.utils.game2seq.typing import (
    Metadata,
    MsgHistoryList,
    OrderSequencePart,
    OrderPredictionOutput,
    AllOrderIndependentPredictionOutput,
    PhaseMsgs,
)
from parlai_diplomacy.utils.game2seq.base_prediction import BaseDiplomacyPredictionFormatter
from parlai_diplomacy.utils.game2seq.dialogue_prediction import (
    DialoguePredictionFormatter,
    TrainingDialoguePredictionFormatter,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    format_player_prompt_token,
    modify_input_prompt_for_power,
    add_recipient_to_prompt_token,
    COUNTRY_ID_TO_POWER,
)

TWO_POWERS_DIALOGUE_FLAG = "two_powers_dialogue"


class _BaseOrderPredictionFormatter(BaseDiplomacyPredictionFormatter, ABC):
    """
    Helper class to convert a dipCC game object to an order sequence prediction task
    for a ParlAI model.
    """

    @abstractmethod
    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata=None
    ) -> Optional[str]:
        """
        Given a game JSON, phase, speaker, and metadata, return the order string
        to predict [optionally]. If None, example will be discarded.

        Must be defined by child classes
        """
        pass

    def get_format_parts(self, fmt: str) -> List[OrderSequencePart]:
        """
        Override from parent class to return format parts.
        """
        if fmt == "speaker_token":
            format_parts = []
        elif fmt == "dummy_token":
            format_parts = [OrderSequencePart.DUMMY_TOKEN]
        elif fmt == "state":
            format_parts = [OrderSequencePart.STATE]
        elif fmt == "message_history":
            format_parts = [OrderSequencePart.MESSAGE_HISTORY]
        elif fmt == "message_history_state":
            format_parts = [OrderSequencePart.MESSAGE_HISTORY, OrderSequencePart.STATE]
        elif fmt == "message_history_orderhistorysincelastmovementphase_state":
            format_parts = [
                OrderSequencePart.MESSAGE_HISTORY,
                OrderSequencePart.ORDER_HISTORY_SINCE_LAST_MOVEMENT,
                OrderSequencePart.STATE,
            ]
        elif fmt == "orderhistorysincelastmovementphase_state":
            format_parts = [
                OrderSequencePart.ORDER_HISTORY_SINCE_LAST_MOVEMENT,
                OrderSequencePart.STATE,
            ]
        elif fmt == "message_history_orderhistorysincelastmovementphase":
            format_parts = [
                OrderSequencePart.MESSAGE_HISTORY,
                OrderSequencePart.ORDER_HISTORY_SINCE_LAST_MOVEMENT,
            ]
        else:
            raise RuntimeError(f"Format {fmt} not currently supported")

        return format_parts

    def _collect_input_format_part_strs(
        self,
        speaker: Power,
        phase: Phase,
        game_json: GameJson,
        metadata: Metadata,
        format_parts: List[OrderSequencePart],
        flattened_game_state: str,
    ) -> List[str]:
        """
        Collect all designated input format part strings for a speaker in a given phase, given a list of format parts
        """
        formatted_input_parts: List[str] = []
        # format input sequence
        for fmt_part in format_parts:
            if fmt_part == OrderSequencePart.STATE:
                formatted_input_parts.append(flattened_game_state)
            elif fmt_part == OrderSequencePart.DUMMY_TOKEN:
                formatted_input_parts.append("UNK")
            elif fmt_part == OrderSequencePart.MESSAGE_HISTORY:
                truncation = metadata["opt"].get("message_history_truncation", 2048)
                filtered_message_history = self.messagehistory_builder.extract_message_history_from_game_json(
                    game_json, phase, speaker, truncation=truncation,
                )
                if metadata.get(TWO_POWERS_DIALOGUE_FLAG):
                    power1, power2 = metadata[TWO_POWERS_DIALOGUE_FLAG]
                    filtered_message_history = _limit_msg_history_to_powers(
                        filtered_message_history, [power1, power2]
                    )

                last_timestamp = get_last_timestamp_msg_history(filtered_message_history)
                draw_state = get_gamejson_draw_state(game_json, last_timestamp, metadata)
                if (
                    metadata["opt"].get("train_on_message_prefixes")
                    and metadata.get("is_training")
                    and filtered_message_history
                    and filtered_message_history[-1]
                ):
                    n_messages_to_keep = random.randint(0, len(filtered_message_history[-1]))
                    filtered_message_history[-1] = filtered_message_history[-1][
                        :n_messages_to_keep
                    ]

                flattened_message_history = self.messagehistory_flattener.flatten_message_history(
                    filtered_message_history, draw_state, metadata=metadata,
                )
                formatted_input_parts.append(flattened_message_history)
            elif fmt_part == OrderSequencePart.ORDER_HISTORY_SINCE_LAST_MOVEMENT:
                order_history_dct = order_helper.build_order_history_dct(game_json, phase)
                order_history_since_last_movement_phase = self.orders_flattener.flatten_order_history_since_last_movement_phase(
                    order_history_dct
                )
                formatted_input_parts.append(order_history_since_last_movement_phase)
            else:
                raise RuntimeError(f'Unrecognized item "{fmt_part}" in order format request.')

        return formatted_input_parts

    def _format_phase_speaker_examples(
        self,
        speaker: Power,
        phase: Phase,
        game_json: GameJson,
        metadata: Metadata,
        format_parts: List[OrderSequencePart],
        flattened_game_state: str,
    ) -> Dict[str, str]:
        """
        Format examples for a speaker in a particular phase.

        Returns a dict of:
        {
            "input": str_input,
            "output": str_output,
        }
        for a given speaker during a phase
        """
        formatted_input_parts = []
        # set up speaker
        phase_seqs_speaker = {}

        # format input sequence
        # (1) Collect all stringified format parts for a given speaker in a phase
        formatted_input_parts: List[str] = self._collect_input_format_part_strs(
            speaker, phase, game_json, metadata, format_parts, flattened_game_state
        )
        # (2) Format the prompt for the given player
        player_prompt_token = format_player_prompt_token(phase, speaker, metadata)
        formatted_input_parts.append(player_prompt_token)
        # (3) Combine to form an input sequence
        phase_seqs_speaker["input"] = f"{self.delimiter}".join(formatted_input_parts)

        # format output sequence, if the order already exists
        orders = self.get_phase_orders_for_speaker(game_json, phase, speaker, metadata)
        if orders is not None:
            phase_seqs_speaker["output"] = orders

        return phase_seqs_speaker

    def generate_input_output_pairs(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[OrderSequencePart]
    ) -> OrderPredictionOutput:
        """
        Generates the order input sequence in the requested format

        Given an ordered list of format_parts, formats the input sequence according to
        these parts as well as any additional options in the metadata.

        For example, if format_parts = [OrderSequencePart.MESSAGE_HISTORY, OrderSequencePart.STATE],
        the input sequence becomes f"{message_history} {state} {player_prompt_token}"
        """
        seqs = {}
        for phase in game_json:
            seqs[phase] = {}
            state = self.state_flattener.flatten_state(
                game_json[phase]["state"],
                phase,
                short_version=metadata.get("shortstate"),
                opt=metadata["opt"],
            )
            for _, speaker in COUNTRY_ID_TO_POWER.items():
                seqs[phase][speaker] = self._format_phase_speaker_examples(
                    speaker, phase, game_json, metadata, format_parts, state
                )

        return seqs


class OrderPredictionFormatter(_BaseOrderPredictionFormatter):
    """
    Helper class to convert a dipCC game object to a sequence format for a ParlAI model.
    """

    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata=None
    ) -> Optional[str]:
        """
        Return flattened orders for a given player.
        """
        if (
            game_json[phase].get("orders") is not None
            and game_json[phase]["orders"].get(speaker) is not None
        ):
            orders = filter_all_holds(game_json, speaker, phase, metadata)
            if orders is not None:
                return self.orders_flattener.flatten_action(orders)

        return None


class OrderRolloutPredictionFormatter(_BaseOrderPredictionFormatter):
    """
    Helper class to convert a dipCC game object to a sequence of orders rolled out
    to the next movement phase.
    """

    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata,
    ) -> Optional[str]:
        """
        Return flattened rolled orders for a given player.
        """
        phases_to_include = order_helper.get_phases_until_next_movement_phase(
            phase,
            game_json,
            including_current=metadata["opt"].get("rollout_except_movement", True),
        )
        flattened_orders = []
        for i, next_phase in enumerate(phases_to_include):
            orders = filter_all_holds(game_json, speaker, next_phase, metadata)
            if orders is None:
                if i == 0:
                    # return None if no speaker orders for this phase
                    return None
                else:
                    # stop appending orders
                    break

            flattened_orders.append((next_phase, self.orders_flattener.flatten_action(orders),))

        return "\n".join(
            [f"{next_phase}\n{flat_orders}" for next_phase, flat_orders in flattened_orders]
        )


class _BaseAllOrderPredictionFormatter(_BaseOrderPredictionFormatter, ABC):
    @staticmethod
    def _get_all_order_dict_all_holds(
        game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata,
    ) -> Tuple[Optional[JointAction], Optional[Dict[str, bool]]]:
        """
        Helper method which returns a dict mapping from power ID (str) to action as well
        as a dict mapping from power ID to a bool indicating whether or not the corresponding
        order would qualify as an "all holds" move.
        """
        all_orders = {}
        all_holds = {}
        if not game_json[phase].get("orders"):
            # sometimes the orders are completely empty
            # we should get rid of these examples
            return None, None

        for player in POWERS:
            orders = game_json[phase]["orders"].get(player, [])
            all_orders[player] = orders

            # now check if all holds
            filtered = filter_all_holds(game_json, player, phase, metadata)
            all_holds[player] = True if filtered is None else False

        return all_orders, all_holds


class AllOrderPredictionFormatter(_BaseAllOrderPredictionFormatter):
    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata
    ) -> Optional[str]:
        """
        Return orders for all powers, with the speaker's orders last in terms of order.
        """
        all_orders, all_holds = self._get_all_order_dict_all_holds(
            game_json, phase, speaker, metadata
        )
        if all_orders is not None and not all_holds[speaker]:
            return self.orders_flattener.flatten_joint_action(
                all_orders,
                speaker,
                mark_all_holds=metadata["opt"].get("allorders_mark_all_holds", False),
                all_holds_dct=all_holds,
            )

        return None


class AllOrderIndependentPredictionFormatter(_BaseAllOrderPredictionFormatter):
    """
    In "all order many" prediction, we predict the orders for all powers from
    the perspective of a single power; however -- unlike in "all order"
    prediction -- we predict each power's orders separately.
    """

    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata
    ):
        """
        Override from parent teacher to return a list of orders instead of
        flattened/stringified orders.
        """
        # filter every order at this stage
        all_orders, all_holds = self._get_all_order_dict_all_holds(
            game_json, phase, speaker, metadata
        )
        if all_orders is None:
            return all_orders

        # remove all holds for each power
        for power, all_holds in all_holds.items():
            if all_holds:
                all_orders[power] = None

        return all_orders

    def _generate_input_output_pairs_maybe_two_powers(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[OrderSequencePart],
    ) -> AllOrderIndependentPredictionOutput:
        """
        Produce per-power order training examples. If two_powers_dialogue is
        set, will generate only examples for these powers.
        """
        two_powers: Optional[Sequence[Power]] = metadata.get(TWO_POWERS_DIALOGUE_FLAG)
        seqs = super().generate_input_output_pairs(game_json, metadata, format_parts)
        for phase, phase_data in seqs.items():
            for speaker, speaker_data in phase_data.items():
                new_speaker_data = []
                output_dct = speaker_data.get("output")
                for power in POWERS:
                    if two_powers is not None and (
                        power not in two_powers or speaker not in two_powers
                    ):
                        continue
                    new_example_data = {
                        "input": modify_input_prompt_for_power(
                            speaker_data["input"], power, self.version
                        ),
                    }
                    if output_dct is not None and output_dct.get(power) is not None:
                        new_example_data["output"] = self.orders_flattener.flatten_action(
                            output_dct[power]
                        )
                    new_speaker_data.append(new_example_data)
                seqs[phase][speaker] = new_speaker_data

        return seqs

    def generate_input_output_pairs(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[OrderSequencePart]
    ) -> AllOrderIndependentPredictionOutput:
        """
        Override from parent class to return a list of examples, instead of a
        single example for all powers order prediction. This function will also
        create two-power-view exmaples if --train_two_powers_view_orders if set.
        """
        seqs = self._generate_input_output_pairs_maybe_two_powers(
            game_json, metadata, format_parts
        )
        if metadata["opt"].get("train_two_powers_view_orders") and metadata.get("is_training"):
            for power1 in POWERS:
                for power2 in POWERS:
                    if power1 != power2:
                        with _dict_override(
                            metadata, TWO_POWERS_DIALOGUE_FLAG, (power1, power2)
                        ) as meta_two_powers:
                            more_seqs = self._generate_input_output_pairs_maybe_two_powers(
                                game_json, meta_two_powers, format_parts
                            )
                        for phase in more_seqs:
                            for speaker in more_seqs[phase]:
                                seqs[phase][speaker].extend(more_seqs[phase][speaker])
        return seqs


class AllOrderRolloutPredictionFormatter(_BaseAllOrderPredictionFormatter):
    """
    For the All Order Rollout teachers, we rollout all orders up until the next movement phase.
    """

    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata
    ) -> Optional[str]:
        """
        Return orders for all powers, with the speaker's orders last in terms of order.
        """

        rollout_joint_action, all_holds_dcts = get_rollout_joint_action_from_game_json(
            game_json, phase, speaker, metadata
        )
        if rollout_joint_action is None:
            return None
        return self.orders_flattener.flatten_rollout_joint_action(
            rollout_joint_action,
            speaker,
            metadata["opt"].get("allorders_mark_all_holds", False),
            all_holds_dcts,
        )


class AllOrderIndependentRolloutPredictionFormatter(_BaseAllOrderPredictionFormatter):
    """
    AllOrderIndependent + AllOrderRollout
    """

    def get_phase_orders_for_speaker(
        self, game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata
    ) -> Optional[RolloutJointAction]:
        """
        Return orders for all powers, with the speaker's orders last in terms of order.
        """
        rollout_joint_action, _ = get_rollout_joint_action_from_game_json(
            game_json, phase, speaker, metadata
        )
        return rollout_joint_action

    def _generate_input_output_pairs_maybe_two_powers(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[OrderSequencePart]
    ) -> AllOrderIndependentPredictionOutput:
        two_powers: Optional[Sequence[Power]] = metadata.get(TWO_POWERS_DIALOGUE_FLAG)
        seqs = super().generate_input_output_pairs(game_json, metadata, format_parts)
        for phase, phase_data in seqs.items():
            for speaker, speaker_data in phase_data.items():
                new_speaker_data = []
                rollout_joint_action = speaker_data.get("output")
                for power in POWERS:
                    if two_powers is not None and (
                        power not in two_powers or speaker not in two_powers
                    ):
                        continue
                    new_example_data = {
                        "input": modify_input_prompt_for_power(
                            speaker_data["input"], power, self.version
                        ),
                    }
                    if (
                        rollout_joint_action is not None
                        and phase in rollout_joint_action
                        and rollout_joint_action[phase].get(power) is not None
                    ):
                        new_example_data["output"] = self.orders_flattener.flatten_rollout_action(
                            order_helper.extract_rollout_action_for_power(
                                rollout_joint_action, power
                            )
                        )
                    new_speaker_data.append(new_example_data)
                seqs[phase][speaker] = new_speaker_data
        return seqs

    def generate_input_output_pairs(
        self, game_json: GameJson, metadata: Metadata, format_parts: List[OrderSequencePart]
    ) -> AllOrderIndependentPredictionOutput:
        """
        Override from parent class to return a list of examples, instead of a
        single example for all powers order prediction.
        """
        seqs = self._generate_input_output_pairs_maybe_two_powers(
            game_json, metadata, format_parts
        )
        if metadata["opt"].get("train_two_powers_view_orders") and metadata.get("is_training"):
            for power1 in POWERS:
                for power2 in POWERS:
                    if power1 != power2:
                        with _dict_override(
                            metadata, TWO_POWERS_DIALOGUE_FLAG, (power1, power2)
                        ) as meta_two_powers:
                            more_seqs = self._generate_input_output_pairs_maybe_two_powers(
                                game_json, meta_two_powers, format_parts
                            )
                        for phase in more_seqs:
                            for speaker in more_seqs[phase]:
                                seqs[phase][speaker].extend(more_seqs[phase][speaker])
        return seqs


class PlausiblePseudoOrderPredictionFormatter(DialoguePredictionFormatter):
    def _maybe_edit_prompt_token(
        self,
        curr_prompt_token: str,
        sender: Power,
        recipient: Optional[Power],
        metadata: Metadata,
    ) -> str:
        """
        Override from dialogue formatter to add the sender
        """
        if recipient is None:
            raise RuntimeError(
                "Must specify recipient for plausible pseudo order prediction model."
            )
        return add_recipient_to_prompt_token(
            curr_prompt_token, sender, recipient, metadata, self.version
        )


class TrainingPlausiblePseudoOrderPredictionFormatter(TrainingDialoguePredictionFormatter):
    def _get_output_str(
        self,
        partial_history_output: PhaseMsgs,
        speaker: Power,
        recipient: Power,
        phase: Phase,
        metadata: Metadata,
        ind: int,
    ) -> str:
        """
        Override the dialogue target output to be the pseudo orders target output
        """
        flattened_powermajor = self._get_train_pseudo_orders(
            speaker,
            recipient,
            phase,
            metadata,
            ind,
            partial_history_output[0][MessageObjectPart.TIME_SENT],
        )
        if not flattened_powermajor:
            # Corrupted
            return flattened_powermajor
        opt = metadata["opt"]
        if opt.get("rollout_phasemajor"):
            assert opt.get("rollout_pseudo_orders")
            assert opt.get("single_view_pseudo_orders")
            joint_action_bilateral = self.orders_unflattener.unflatten_rollout_joint_action_bilateral_powermajor(
                flattened_powermajor, phase
            )
            return self.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                joint_action_bilateral,
                speaker,
                recipient,
                speaker_first=opt.get("speaker_first", False),
            )
        else:
            return flattened_powermajor

    def _maybe_edit_prompt_token(
        self, curr_prompt_token: str, sender: Power, recipient: Power, metadata: Metadata
    ) -> str:
        """
        Override from dialogue formatter to add the recipient to the prompt token
        """
        return add_recipient_to_prompt_token(
            curr_prompt_token, sender, recipient, metadata, self.version
        )


def get_rollout_joint_action_from_game_json(
    game_json: GameJson, phase: Phase, speaker: Power, metadata: Metadata
) -> Tuple[Optional[RolloutJointAction], Dict[Phase, Dict[Power, bool]]]:
    rollout_joint_action = {}
    all_holds_dcts = {}

    # first check that orders for this phase even exist, otherwise we can assume we are inference time
    if not game_json[phase].get("orders"):
        return None, all_holds_dcts

    # next, find the next movement phase provided a list of all phases
    phases_to_include = order_helper.get_phases_until_next_movement_phase(phase, game_json)
    for next_phase in phases_to_include:
        all_orders, all_holds = _BaseAllOrderPredictionFormatter._get_all_order_dict_all_holds(
            game_json, next_phase, speaker, metadata
        )
        if all_orders is None or all_holds[speaker]:
            # If the next phase of orders is None or All Holds for the speaker,
            # We return None
            return None, all_holds_dcts
        elif all_orders is not None:
            rollout_joint_action[next_phase] = all_orders
            all_holds_dcts[next_phase] = all_holds

    return rollout_joint_action, all_holds_dcts


def filter_all_holds(game_json, speaker: str, phase: str, metadata=None) -> Optional[Action]:
    """
    Filters all holds if conditions are met.

    Returns None if orders contain all holds.
    """
    orders = game_json[phase]["orders"].get(speaker, [])

    if metadata is None or not metadata.get("filter_all_holds"):
        # return orders as-is
        return orders

    def get_num_units() -> int:
        # return the number of orderable units
        if phase.endswith("M"):
            # movement phase
            return len(game_json[phase]["state"]["units"][speaker])
        elif phase.endswith("R"):
            # retreat phase
            return len(game_json[phase]["state"]["retreats"][speaker])
        elif phase.endswith("A"):
            # build phase
            builds = game_json[phase]["state"]["builds"][speaker]
            build_count = int(builds["count"])
            if build_count >= 0:  # build
                return min(build_count, len(builds["homes"]))
            else:  # disband
                return -build_count
        else:
            raise RuntimeError(f"Incorrect phase: {phase}")

    num_units = get_num_units()
    # missing orders
    if len(orders) < num_units:
        if not phase.endswith("A"):
            return None
    # all holds
    if phase.endswith("M") and num_units >= 3 and all(x.endswith(" H") for x in orders):
        return None

    return orders


def _limit_msg_history_to_powers(
    msg_history: MsgHistoryList, powers: List[Power]
) -> MsgHistoryList:
    powers_set = frozenset(powers)
    copied = []
    for phase_message in msg_history:
        copied.append([])
        for msg in phase_message:
            if powers_set == frozenset([msg["sender"], msg["recipient"]]):
                copied[-1].append(msg)
    return copied


@contextlib.contextmanager
def _dict_override(d: Dict, key: str, value):
    d = d.copy()
    d[key] = value
    yield d
