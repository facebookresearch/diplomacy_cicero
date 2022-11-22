#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC
import logging
from typing import Dict, Tuple, List, Any, Optional

from parlai_diplomacy.utils.game2seq.format_helpers.misc import (
    ParlaiDecodingError,
    get_input_format,
    get_output_type,
    remove_end_token,
)
from parlai_diplomacy.utils.game2seq.format_helpers.opt_utils import expects_recipient
import parlai_diplomacy.utils.misc as misc
from parlai_diplomacy.utils.token_metadata import clean_token_metadata
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.typing import Metadata, ParlAIAct
from parlai_diplomacy.utils.game2seq.factory import sequence_formatter_factory
from parlai_diplomacy.utils.game2seq import input_validation
from fairdiplomacy.viz.meta_annotations import api as meta_annotations

from fairdiplomacy.typedefs import (
    MessageDict,
    OutboundMessageDict,
    Power,
    Phase,
    Timestamp,
)
from fairdiplomacy.utils.typedefs import build_outbound_message_dict
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.game import assert_game_from_view_of
from fairdiplomacy.pseudo_orders import PseudoOrders

MAX_DIALOGUE_ATTEMPTS = 5  # max no. of times to try to solicit a message
TOKEN_DETAILS_TAG = "token_details"

"""
Module to wrap ParlAI agent to produce orders given a game object.
"""


class BaseDialogueWrapper(BaseWrapper, ABC):
    """
    Base Wrapper for ParlAI agent that produces dialogue.
    """

    OPT_OVERRIDE = {**BaseWrapper.OPT_OVERRIDE, "verbose": True}

    def _get_player_metadata(self, opt: Dict[Any, Any]) -> Metadata:
        """
        Set metadata for a given player based on the opt
        """
        metadata = super()._get_player_metadata(opt)

        # player personality
        if self.opt.get("include_player_personalities", False):
            raise DeprecationWarning("include_player_personalities is deprecated")

        return metadata

    def expects_recipient(self) -> bool:
        return expects_recipient(self.opt)

    def set_block_redacted_tokens(self) -> None:
        """
        Invoke this function to block redacted tokens.

        Only needs to be called once. Cannot be undone without re-creating the model.
        """
        for tok in [" [", "[", '"[', ' "[', "([", " ([", "'[", " '[", "[REDACTED]"]:
            self.add_literal_to_block_list(tok)

    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Given a game object, return a dictionary of formatted input sequences for each
        power to feed directly to the model

        Args:
            game: dipcc game object
            view_of_power: (str) power to get input for
            target_power: optional power; used when we have a recipient classifier in-the-loop

        Returns:
            A dict of `power` -> `input sequence for model`
        """
        format_output = self.formatter.change_format(
            game,
            self.input_format_str,
            self.metadata,
            view_of_power,
            recipient=target_power,
            timestamp=timestamp,
        )
        _, seqs = misc.last_dict_item(format_output)
        seq = seqs["input"]

        return seq

    def format_output_seq(self, output_seq: str, power: Power,) -> List[OutboundMessageDict]:
        phase = self.game.get_current_phase()
        return self.formatter.messagehistory_unflattener.unflatten_model_output_messages(
            output_seq, power, phase
        )

    def _get_prefix_to_power(self, game: pydipcc.Game, power: str, to_power: str,) -> str:
        """
        get the prefix of messages from one power to a particular power.

        - game: Game object
        - power: Sender
        - to_power: Recipient power

        return:
            prefix str should look something like f"{phase}\n{power} -> {to_power}:"
        """
        msg_dct: MessageDict = {
            MessageObjectPart.SENDER: power,
            MessageObjectPart.RECIPIENT: to_power,
            MessageObjectPart.MESSAGE: "",
            MessageObjectPart.PHASE: game.current_short_phase,
            MessageObjectPart.TIME_SENT: Timestamp.from_centis(0),
        }
        # prefix str should look something like f"{phase}\n{power} -> {to_power}:"
        prefix_str = (
            self.formatter.messagehistory_flattener.flatten_phase_messages([msg_dct],).split(":")[
                0
            ]
            + ":"
        )

        return prefix_str

    def produce_messages(
        self,
        game: pydipcc.Game,
        power: str,
        timestamp: Timestamp,
        recipient: Power = None,
        prefix_str: Optional[str] = None,
    ) -> List[OutboundMessageDict]:
        self.game = game

        seq = self.format_input_seq(game, power, recipient, timestamp)
        logging.debug(f"\n(ParlAIDialogueWrapper) Input sequence:\n{seq}\n")

        msgs, pred_details = [], None
        for i in range(MAX_DIALOGUE_ATTEMPTS):
            raw_pred = self.get_model_pred(seq, prefix_str=prefix_str)
            raw_pred_text = raw_pred["text"]

            logging.debug(f"\n(ParlAIDialogueWrapper) Output sequence:\n{raw_pred_text}\n")
            try:
                msgs = self.format_output_seq(raw_pred_text, power)
            except ParlaiDecodingError as e:
                # Once in a while we get badly-formatted messages. Watcha gonna do?
                logging.warning(f"{e}")
            if msgs:
                # We successfully generated a message
                if "text_token_info" in raw_pred:
                    pred_details = clean_token_metadata(raw_pred["text_token_info"])
                    meta_annotations.add_dict_next_msg(
                        {"data": pred_details}, tag=TOKEN_DETAILS_TAG
                    )
                break
            elif i < MAX_DIALOGUE_ATTEMPTS - 1:
                logging.warning(f"Sampled a bad message: {raw_pred}; trying again")
            else:
                logging.error(f"Tried {MAX_DIALOGUE_ATTEMPTS} to sample a good message but failed")

        return msgs

    def produce_many_messages(
        self,
        game: pydipcc.Game,
        power: Power,
        timestamp: Timestamp,
        num_preds: int,
        recipient: Power = None,
        batch_size: Optional[int] = None,
        prefix_str: Optional[str] = None,
    ) -> List[Tuple[List[OutboundMessageDict], float]]:
        """
        Produce multiple orders by beam search.
        """
        self.game = game
        seq = self.format_input_seq(game, power, target_power=recipient, timestamp=timestamp)
        logging.debug(f"\n(ParlAIDialogueWrapper) Input sequence:\n{seq}\n")
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size, prefix_str=prefix_str)
        logging.debug(f"\n(ParlAIDialogueWrapper) Output sequence:\n{many_raw_pred}\n")

        try:
            message_scores = [
                (self.format_output_seq(raw_pred, power), float(score))
                for raw_pred, score in many_raw_pred
            ]
            return message_scores
        except ParlaiDecodingError as e:
            # Once in a while we get badly-formatted messages. Watcha gonna do?
            logging.warning(f"{e}")
            return []

    def produce_messages_to_power_prefix(
        self, game: pydipcc.Game, power: str, timestamp: Timestamp, to_power: str,
    ) -> List[OutboundMessageDict]:
        """
        Produce messages from one power to a particular power. This works by forcing
        the model to decode a set of prefix tokens, and then continuing generation.

        - game: Game object
        - power: Sender
        - to_power: Recipient power
        """
        prefix_str = self._get_prefix_to_power(game=game, power=power, to_power=to_power)
        return self.produce_messages(game, power, timestamp, prefix_str=prefix_str)

    def produce_many_messages_to_power_prefix(
        self,
        game: pydipcc.Game,
        power: str,
        timestamp: Timestamp,
        to_power: str,
        num_preds: int,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[List[OutboundMessageDict], float]]:
        """
        Produce many messages from one power to a particular power. This works by forcing
        the model to decode a set of prefix tokens, and then continuing generation.

        - game: Game object
        - power: Sender
        - to_power: Recipient power
        """
        prefix_str = self._get_prefix_to_power(game=game, power=power, to_power=to_power)
        return self.produce_many_messages(
            game,
            power,
            timestamp,
            num_preds,
            recipient=None,
            batch_size=batch_size,
            prefix_str=prefix_str,
        )

    def format_candidate_seqs(
        self,
        candidates: List[str],
        sender: Power,
        recipient: Power,
        phase: Phase,
        add_eom_token: bool = True,
    ) -> List[str]:
        candidate_seqs = [
            self.formatter.messagehistory_flattener.flatten_outbound_message_candidate(
                build_outbound_message_dict(sender, recipient, msg, phase)
            )
            for msg in candidates
        ]
        if not add_eom_token:
            candidate_seqs = [remove_end_token(seq, "[EO_M]") for seq in candidate_seqs]
        return candidate_seqs

    def get_candidate_logprobs(
        self,
        act: Dict[str, Any],
        skip_prefix: Optional[bool] = None,
        skip_end_token: Optional[bool] = None,
    ) -> List[float]:
        # Call BaseWrapper method but with default skip_* == True
        return super().get_candidate_logprobs(
            act,
            skip_prefix=(skip_prefix if skip_prefix is not None else True),
            skip_end_token=(skip_end_token if skip_end_token is not None else True),
        )

    def score_candidate_messages(
        self,
        game: pydipcc.Game,
        candidates: List[str],
        sender: Power,
        timestamp: Optional[Timestamp],
        recipient: Power,
        add_eom_token: bool = True,
        skip_prefix: bool = True,
        skip_end_token: bool = True,
    ) -> List[Tuple[str, float]]:
        self.game = game
        phase = game.current_short_phase
        candidate_seqs = self.format_candidate_seqs(
            candidates, sender, recipient, phase, add_eom_token=add_eom_token,
        )
        if not add_eom_token:
            assert (
                skip_end_token
            ), "[EO_M] token is not added, so __end__ token should be skipped, otherwise, the prob will be biased"
        candidate_seqs_and_logps = self.score_candidate_seqs(
            game, candidate_seqs, sender, recipient, timestamp, skip_prefix, skip_end_token,
        )

        return [
            (self.format_output_seq(candidate_seq, sender)[0]["message"], logp,)
            for candidate_seq, logp in candidate_seqs_and_logps
        ]


class ParlAIDialogueWrapper(BaseDialogueWrapper):
    """
    Wrapper for ParlAI agent that produces dialogue from traditional-view.
    i.e., given bot A's view of the game, generate replies from bot A to other players.
    This should be used for most cases.
    """


class ParlAIResponseViewDialogueWrapper(BaseDialogueWrapper):
    """
    Wrapper for ParlAI agent that produces dialogue from response-view,
    i.e., given bot A's view of the game, generate replies from other players to bot A.
    This is useful for the zero-shot nonsense classifier during interactive games,
    since we can only use our own view of the game and cannot leak other bot's view of the game.
    """

    def score_candidate_messages(
        self,
        game: pydipcc.Game,
        candidates: List[str],
        sender: Power,
        timestamp: Optional[Timestamp],
        recipient: Power,
        add_eom_token: bool = True,
        skip_prefix: bool = True,
        skip_end_token: bool = True,
    ) -> List[Tuple[str, float]]:
        # The game view is from the recipient, instead of the sender
        self.game = game
        assert_game_from_view_of(game, recipient)
        phase = game.current_short_phase
        candidate_seqs = self.format_candidate_seqs(
            candidates, sender, recipient, phase, add_eom_token=add_eom_token,
        )
        if not add_eom_token:
            assert (
                skip_end_token
            ), "[EO_M] token is not added, so __end__ token should be skipped, otherwise, the prob will be biased"
        # The game view is from the recipient, instead of the sender
        candidate_seqs_and_logps = self.score_candidate_seqs(
            game=game,
            candidate_seqs=candidate_seqs,
            view_of_power=recipient,  # view of power is from the recipient, because this is response-view
            target_power=None,  # only necessary for formatting bilateral pseudo-order inputs, but response-view dialogue teacher is never trained using pseudo-order
            timestamp=timestamp,
            skip_prefix=skip_prefix,
            skip_end_token=skip_end_token,
        )
        return [
            (self.format_output_seq(candidate_seq, sender)[0]["message"], logp,)
            for candidate_seq, logp in candidate_seqs_and_logps
        ]

    def produce_response_view_messages(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        timestamp: Timestamp,
        response_from_power: Power,
    ) -> List[OutboundMessageDict]:
        """
        Produce message from response-view,
        i.e., bot A's view of the game --> message from other players to bot A

        game: Game object from the view of `view_of_power`
        view_of_power: the game view should be from this power
        response_from_power: the response is from this power
        """
        assert_game_from_view_of(game, view_of_power)
        self.game = game
        # The game view is from the recipient (view_of_power), instead of the message sender
        seq = self.format_input_seq(game, view_of_power, target_power=None, timestamp=timestamp)

        prefix_str = self._get_prefix_to_power(
            game=game, power=response_from_power, to_power=view_of_power
        )
        logging.debug(f"\n(ParlAIResponseViewDialogueWrapper) Input sequence:\n{seq}\n")
        raw_pred = self.get_model_pred(seq, prefix_str=prefix_str)["text"]
        logging.debug(f"\n(ParlAIResponseViewDialogueWrapper) Output sequence:\n{raw_pred}\n")

        try:
            return self.format_output_seq(raw_pred, response_from_power)
        except ParlaiDecodingError as e:
            # Once in a while we get badly-formatted messages. Watcha gonna do?
            logging.warning(f"{e}")
            return []

    def produce_many_response_view_messages(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        timestamp: Timestamp,
        response_from_power: Power,
        num_preds: int,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[List[OutboundMessageDict], float]]:
        """
        Produce multiple messages from response view by beam search.
        i.e., bot A's view of the game --> many message from other players to bot A

        game: a game view from `view_of_power`
        view_of_power: the game view should be for this power
        response_from_power: the response is from this power
        """
        assert_game_from_view_of(game, view_of_power)
        self.game = game
        # The game view is from the recipient (to_power), instead of the sender
        seq = self.format_input_seq(game, view_of_power, target_power=None, timestamp=timestamp)

        prefix_str = self._get_prefix_to_power(
            game=game, power=response_from_power, to_power=view_of_power
        )
        logging.debug(f"\n(ParlAIResponseViewDialogueWrapper) Input sequence:\n{seq}\n")
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size, prefix_str=prefix_str)
        logging.debug(f"\n(ParlAIResponseViewDialogueWrapper) Output sequence:\n{many_raw_pred}\n")

        try:
            message_scores = [
                (self.format_output_seq(raw_pred, response_from_power), score)
                for raw_pred, score in many_raw_pred
            ]
            return message_scores
        except ParlaiDecodingError as e:
            # Once in a while we get badly-formatted messages. Watcha gonna do?
            logging.warning(f"{e}")
            return []

    def produce_messages_to_power_prefix(
        self, game: pydipcc.Game, power: str, timestamp: Timestamp, to_power: str,
    ):
        raise ValueError(
            """
            `produce_messages_to_power_prefix` shouldn't be called in Response-view teacher,
            since the response should always be to the power whose view is used
            """
        )

    def produce_many_messages_to_power_prefix(
        self,
        game: pydipcc.Game,
        power: str,
        timestamp: Timestamp,
        to_power: str,
        num_preds: int,
        batch_size: Optional[int] = None,
        prefix_str: Optional[str] = None,
    ):
        raise ValueError(
            """
            `produce_many_messages_to_power_prefix` shouldn't be called in Response-view teacher,
            since the response should always be to the power whose view is used
            """
        )
