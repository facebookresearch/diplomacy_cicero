#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.timestamp import Timestamp
from parlai.utils import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.typing import MessageDict, Metadata
from parlai_diplomacy.wrappers.base_wrapper import load_opt
from parlai_diplomacy.wrappers.orders import (
    BaseOrderWrapper,
    ParlAIAllOrderWrapper,
    ParlAISingleOrderWrapper,
)
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
from parlai_diplomacy.utils.game2seq.format_helpers.misc import get_output_type
import parlai_diplomacy.utils.misc as misc

from fairdiplomacy import pydipcc
from fairdiplomacy.typedefs import Action, Power, JointAction, RolloutAction
from fairdiplomacy.utils.game import game_from_two_party_view, game_from_view_of


"""
Wrapper for annotated pseudo orders models.

This is separate from orders models, as these wrappers are *only* used to compute
annotated pseudo orders, and not to predict orders in a game setting. In other words,
this wrapper is used to mimic how pseudo orders are used in the training set.

This wrapper is for DEBUG/DEV ONLY.

At train time we (typically, as of 10/5/21) use:

* Use a truthful model
* Use an affirmative reply
* Showing the model the “future” message
* And... in the single-view case, pseudo orders for the self and partner are annotated separately(e.g., the partner is only shown the 2-person history)

We use the TrainingDialoguePrediction formatter here, because a lot of this functionality
was baked into that formatter when we used the pseudo orders to annotate the dialogue training set.
"""


class DevOnlyBaseAnnotatedPseudoOrdersWrapper(BaseOrderWrapper):
    OPT_OVERRIDE = {
        "interactive_mode": True,
        "skip_generation": False,
        "beam_length_penalty": 0,  # For orders models, we turn off length penalties
        "set_player_rating": 5,
        "pseudo_order_generation_inject_all": True,
    }

    def _initialize_formatter(self):
        # pseudo order generation is formulated as a dialogue prediction task
        # so the input formatter should be a dialogue formatter
        self.formatter = sequence_formatter_factory(
            self.task_name.replace(get_output_type(self.task_name), "dialogue"),
            self.metadata["task_version"],
            training=True,
        )

    def _get_player_metadata(self, opt) -> Metadata:
        """
        Override from base class in order to set `pseudo_order_gen` to True
        """
        metadata = super()._get_player_metadata(opt)
        # Turn on pseudo order generation
        metadata[
            "pseudo_order_gen"
        ] = True  # this is a flag we use to generate pseudo orders for the train set
        metadata["game_id"] = -1
        return metadata

    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Returns input for model
        """
        format_output = self.formatter.change_format(game, self.input_format_str, self.metadata)
        _, seqs = misc.last_dict_item(format_output)
        power_seqs = seqs[view_of_power]
        return power_seqs[-1]["input"]

    def produce_pseudo_orders_for_phase(
        self,
        game: pydipcc.Game,
        power: Power,
        to_power: Optional[Power] = None,
        rollout: bool = False,
    ) -> Optional[JointAction]:
        """
        Stub for producing pseudo orders for a given power /phase. Child classes must override
        """
        raise NotImplementedError("Must implement this function for pseudo orders wrappers")

    def score_candidate_pseudo_orders(
        self, game: pydipcc.Game, candidates: List[JointAction], msg: MessageDict, *args, **kwargs
    ) -> Dict[Power, List[Tuple[Action, float]]]:
        """
        Score candidate pseudo orders
        """
        raise NotImplementedError(
            "In order to re-score pseudo orders, "
            "child classes must implement `format_candidate_seqs`"
        )


class DevOnlyAnnotatedPseudoAllOrdersWrapper(
    DevOnlyBaseAnnotatedPseudoOrdersWrapper, ParlAIAllOrderWrapper
):
    def __init__(self, *args, **kwargs):
        BaseOrderWrapper.__init__(self, *args, **kwargs)
        self.out_type = get_output_type(self.task_name)
        assert self.out_type in {"allorderrollout", "allorder"}

    def format_output_seq(
        self, raw_pred: str, game: pydipcc.Game, rollout: bool = False
    ) -> Optional[Union[JointAction, Dict[Power, RolloutAction]]]:
        if self.out_type == "allorder":
            assert not rollout
            orders = self.formatter.orders_unflattener.unflatten_joint_action(raw_pred)
        else:
            # all order rollout
            order_dct = self.formatter.orders_unflattener.unflatten_rollout_joint_action(raw_pred)
            if rollout:
                return order_dct
            orders = order_dct.get(game.current_short_phase)

        return orders

    def produce_pseudo_orders_for_phase(
        self,
        game: pydipcc.Game,
        power: Power,
        to_power: Optional[Power] = None,
        rollout: bool = False,
    ) -> Optional[Union[JointAction, Dict[Power, RolloutAction]]]:
        """
        Given a game, produces pseudo orders for power dialogue from the latest
        phase. Returns a list of tuples of pseudo order labels for the power's dialogue
        utterances during that phase.

        Args:
        game: dipCC game object
        power: name of the power for which orders are queried.

        Returns:
        orders: Predicted JointAction
        """
        seq = self.format_input_seq(game, power)
        raw_pred = self.get_model_pred(seq)["text"]
        orders = self.format_output_seq(raw_pred, game, rollout)

        return orders


class DevOnlyAnnotatedPseudoSingleOrdersWrapper(
    DevOnlyBaseAnnotatedPseudoOrdersWrapper, ParlAISingleOrderWrapper
):
    def __init__(self, *args, **kwargs):
        BaseOrderWrapper.__init__(self, *args, **kwargs)
        self.out_type = get_output_type(self.task_name)
        assert self.out_type in {"order", "orderrollout"}

    def format_output_seq(
        self, raw_pred: str, game: pydipcc.Game, rollout: bool = False
    ) -> Optional[Union[Action, Dict[Power, RolloutAction]]]:
        if self.out_type == "order":
            assert not rollout
            orders = self.formatter.orders_unflattener.unflatten_action(raw_pred)
        else:
            orders_dct = self.formatter.orders_unflattener.unflatten_rollout_action(raw_pred)
            if rollout:
                return orders_dct  # type: ignore
            else:
                orders = orders_dct.get(game.current_short_phase)

        return orders

    def _format_selfview_input_seq(self, game: pydipcc.Game, power: Power):
        """
        Format the input to retrieve orders for the SPEAKER
        """
        return self.format_input_seq(game, power)

    def _format_partnerview_input_seq(self, game: pydipcc.Game, power: Power, to_power: Power):
        """
        Format the input to retrieve orders for the PARTNER
        """
        # change to a 2-party game view so as not to "leak" any information
        two_party_game_view = game_from_two_party_view(game, power, to_power)
        other_seq = self.format_input_seq(two_party_game_view, power)
        # HACK: replace the prompt to make it seem like we are prompting the prediction from view of the OTHER power
        power_str = power.capitalize() if self.formatter.version <= 1 else power
        to_power_str = to_power.capitalize() if self.formatter.version <= 1 else to_power
        curr_prompt = f"{game.current_short_phase} {power_str}"
        assert curr_prompt in other_seq
        to_replace_prompt = f"{game.current_short_phase} {to_power_str}"
        other_seq = other_seq.replace(curr_prompt, to_replace_prompt)

        return other_seq

    def produce_pseudo_orders_for_phase(
        self,
        game: pydipcc.Game,
        power: Power,
        to_power: Optional[Power] = None,
        rollout: bool = False,
        self_prefix: Optional[str] = None,
        partner_prefix: Optional[str] = None,
    ) -> Optional[Union[Action, Dict[Power, RolloutAction]]]:
        """
        Given a game, produces pseudo orders for power dialogue from the latest
        phase. Returns a list of tuples of pseudo order labels for the power's dialogue
        utterances during that phase.

        Args:
        game: dipCC game object
        power: name of the power for which orders are queried.
        to_power: recipient
        rollout: whether or not we expect rollout pseudo orders to be returned
        self_prefix: optional -- string prefix to seed decoding for the self prediction
        partner_prefix: optional -- string prefix to seed decoding for the partner prediction

        Returns:
        orders: predicted *partial* JointAction (only predicts orders for `power`, and `to_power`)
        """
        assert game_from_view_of(game, power)
        assert to_power is not None, "Single pseudo orders requires the recipient to be known"

        orders = {}

        # get self prediction
        self_seq = self._format_selfview_input_seq(game, power)
        self_raw_pred = self.get_model_pred(self_seq, prefix_str=self_prefix)["text"]
        orders[power] = self.format_output_seq(self_raw_pred, game, rollout)

        if to_power == power:
            # talk to self, no need to re-compute
            return orders

        # now get prediction from "other" power's perspectives
        other_seq = self._format_partnerview_input_seq(game, power, to_power)
        other_raw_pred = self.get_model_pred(other_seq, prefix_str=partner_prefix)["text"]
        orders[to_power] = self.format_output_seq(other_raw_pred, game, rollout)

        return orders

    def _score_actions_with_input(
        self, input_seq: str, actions: List[Action]
    ) -> List[Tuple[str, float]]:
        candidate_seqs = [
            self.formatter.orders_flattener.flatten_action(action) for action in actions
        ]
        act = self.get_model_pred(input_seq, candidates=candidate_seqs)
        output_logprobs = self.get_candidate_logprobs(act)

        return [(a, score) for a, score in zip(act["text_candidates"], output_logprobs)]

    def _score_selfview_candidate_actions(
        self, game: pydipcc.Game, actions: List[Action], power: Power
    ) -> List[Tuple[str, float]]:
        seq = self._format_selfview_input_seq(game, power)
        logging.debug(f"\nSelf-view pseudo input seq:\n{seq}\n\n")
        return self._score_actions_with_input(seq, actions)

    def _score_partnerview_candidate_actions(
        self, game: pydipcc.Game, actions: List[Action], power: Power, to_power: Power
    ):
        seq = self._format_partnerview_input_seq(game, power, to_power)
        logging.debug(f"\nPartner-view pseudo input seq:\n{seq}\n\n")
        return self._score_actions_with_input(seq, actions)

    def score_candidate_pseudo_orders(
        self,
        game: pydipcc.Game,
        candidates: List[JointAction],
        msg: MessageDict,
        include_future_message: bool = True,
        injected_sentence: Optional[str] = "I've entered those orders.",
    ) -> Dict[Power, List[Tuple[Action, float]]]:
        """
        Score candidate pseudo orders.
        - game: game object
        - candidates: list of joint actions
        - message: message that we are scoring pseudo orders for
        - include_future_message: whether or not to include the message we are calculating pseudo orders;
            this is True by default, but useful to turn off, for example, to get a score for an order
            before the message was sent
        - injected sentence: Optional string used to inject an affirmative reply for the calculation; if
            it is None, no sentence will be injected

        Returns: A dict mapping from power to a list of scored actions for that power
        """
        self.metadata["opt"][
            "pseudo_order_generation_future_message"
        ] = include_future_message  # Change include future message
        self.metadata["opt"][
            "pseudo_order_generation_injected_sentence"
        ] = injected_sentence  # Change injected sentence

        sender = msg[MessageObjectPart.SENDER].upper()
        assert game_from_view_of(game, sender)  # Should be view of sender
        recipient = msg[MessageObjectPart.RECIPIENT].upper()

        self_actions = [action[sender] for action in candidates]
        self_scores = self._score_selfview_candidate_actions(game, self_actions, sender)

        partner_actions = [action[recipient] for action in candidates]
        partner_scores = self._score_partnerview_candidate_actions(
            game, partner_actions, sender, recipient
        )

        return {
            sender: [(action, score[1]) for score, action in zip(self_scores, self_actions)],
            recipient: [
                (action, score[1]) for score, action in zip(partner_scores, partner_actions)
            ],
        }


def build_annotated_pseudo_orders_wrapper(
    model_path: str, overrides: Dict[Any, Any]
) -> DevOnlyBaseAnnotatedPseudoOrdersWrapper:
    """
    Builds and returns a pseudo orders wrapper provided:
    - A model path
    - A dict of overrides
    """
    additional_args = {"overrides": overrides}
    model_opt = load_opt(model_path)
    output_type = get_output_type(model_opt["task"])
    if "allorder" in output_type:
        wrapper = DevOnlyAnnotatedPseudoAllOrdersWrapper(
            model_path=model_path, additional_args=additional_args
        )
    else:
        wrapper = DevOnlyAnnotatedPseudoSingleOrdersWrapper(
            model_path=model_path, additional_args=additional_args
        )

    return wrapper
