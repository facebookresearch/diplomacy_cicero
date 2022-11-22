#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from fairdiplomacy.utils.agent_interruption import raise_if_should_stop
from collections import Counter
from fairdiplomacy.utils.parlai_multi_gpu_wrappers import (
    load_wrapper_executor,
    wrap_parlai_model_to_executor,
)
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper
import json
import random
import re
from typing import Dict, List, Optional, Tuple, cast

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.typedefs import (
    JointAction,
    MessageDict,
    Order,
    Phase,
    Power,
    SleepTimes,
    Timestamp,
)
from fairdiplomacy.agents.parlai_order_handler import (
    BaseOrderWrapper,
    filter_orders_many_powers,
    get_all_possible_orders_by_power,
)
from fairdiplomacy.utils.game import (
    action_has_any_direct_attack,
    is_friendly_xpower_support_or_convoy,
    is_replying_to,
    get_last_message_from,
)
from fairdiplomacy.utils.typedefs import (
    build_draw_vote_message_dict,
    build_undraw_vote_message_dict,
    with_time_sent,
)
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.viz.meta_annotations.annotator import MetaAnnotator

from parlai_diplomacy.tasks.draw_classifier.agents import DrawVoteStatus
from parlai_diplomacy.utils.game2seq.format_helpers.message_editing import (
    MessageEditing,
    MessageFiltering,
)
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
from parlai_diplomacy.utils.game2seq.format_helpers.orders import is_movement_phase
from parlai_diplomacy.utils.misc import last_dict_key
from parlai_diplomacy.utils.game2seq.input_validation import InputValidator
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME
from parlai_diplomacy.wrappers.dialogue import TOKEN_DETAILS_TAG
from parlai_diplomacy.wrappers.factory import (
    load_dialogue_wrapper,
    load_draw_classifier_wrapper,
    load_ensemble_nonsense_classifier_wrapper,
    load_order_wrapper,
    load_recipient_classifier_wrapper,
    load_sleep_classifier_wrapper,
    load_pseudo_orders_wrapper,
)
from parlai_diplomacy.wrappers.dialogue import BaseDialogueWrapper
from parlai_diplomacy.wrappers.orders import ParlAIAllOrderIndependentRolloutWrapper

MAX_ATTEMPTS_GEN_VALID = 5


def game_hash(game: Game) -> Tuple[Phase, int]:
    return (game.current_short_phase, len(game.messages))


class ParlaiMessagePseudoOrdersCache:
    """Holds cached pseudoorder state for implementing
    cfg.reuse_pseudo_for_consecutive_messages and cfg.reuse_pseudo_for_phase

    This cache is pickleable, and should be saved and loaded across preemption of a job
    in order to maintain the correct behavior of the ParlaiMessageHandler.
    """

    def __init__(self):
        self.cached_pseudo: Optional[Tuple[Phase, Timestamp, PseudoOrders, Optional[Power]]] = None

    def maybe_get(
        self,
        game: pydipcc.Game,
        power: Power,
        reuse_pseudo_for_consecutive_messages: bool,
        reuse_pseudo_for_phase: bool,
        recipient: Optional[Power],
    ) -> Optional[PseudoOrders]:
        if self.cached_pseudo is None:
            return None

        phase, last_ts, last_pseudo, last_recipient = self.cached_pseudo
        if phase != game.current_short_phase:
            return None

        if reuse_pseudo_for_phase:
            logging.info(f"Reusing Pseudo orders for {power}: {last_pseudo}")
            return last_pseudo

        if reuse_pseudo_for_consecutive_messages and recipient == last_recipient:
            new_received_messages = [
                m
                for m in game.messages.values()
                if m[MessageObjectPart.TIME_SENT] > last_ts
                and m[MessageObjectPart.RECIPIENT] == power
            ]
            if not new_received_messages:
                logging.info(f"Reusing Pseudo orders for {power}: {last_pseudo}")
                return last_pseudo

        return None

    def set(self, game: Game, pseudo: PseudoOrders, recipient: Optional[Power]):
        ts = last_dict_key(game.messages) if game.messages else Timestamp.from_seconds(-1)
        self.cached_pseudo = (game.current_short_phase, ts, pseudo, recipient)


class SleepSixTimesCache:
    """
    Caches the sleepsix next recipient, sanity checked by a game hash
    """

    def __init__(self):
        self._hash = None
        self._sleep_times: Optional[SleepTimes] = None

    def get_recipient_sleep_time(self, game: pydipcc.Game) -> Tuple[Power, Timestamp]:
        assert self._sleep_times is not None, "Did you forget to call get_sleep_time first?"
        assert game_hash(game) == self._hash, "get_sleep_time was called on a different game state"

        # check all inf sleep
        all_inf_sleep = all(t[0] == INF_SLEEP_TIME for p, t in self._sleep_times.items())
        if not all_inf_sleep:
            # choose target with lowest sleep time as next recipient and return
            # its sleep time
            powers_times = [(t[0], p) for p, t in self._sleep_times.items()]
            random.shuffle(powers_times)  # break ties
            sleep_time, recipient = min(powers_times, key=lambda x: x[0])
        else:
            # Special case: if predicted time is inf for all powers, choose the recipient with
            # the lowest P(inf)
            logging.warning(
                "Predicted inf sleep for all powers; choosing the power with min(P(inf))"
            )
            powers_inf_probs = [(t[1], p) for p, t in self._sleep_times.items()]
            random.shuffle(powers_inf_probs)  # break ties
            _, recipient = min(powers_inf_probs, key=lambda x: x[0])
            sleep_time = INF_SLEEP_TIME

        return recipient, sleep_time

    def should_recompute(self, game: Game) -> bool:
        return self._sleep_times is None or game_hash(game) != self._hash

    def set_sleep_times(self, game: Game, sleep_times: SleepTimes):
        self._hash = game_hash(game)
        self._sleep_times = sleep_times
        logging.info(f"SleepsixTimesCache updated: {sleep_times}")

    def block_messages_to_power(self, game: Game, recipient: Power):
        assert self._sleep_times is not None, "Did you forget to call get_sleep_time first?"
        assert game_hash(game) == self._hash, "get_sleep_time was called on a different game state"
        self._sleep_times[recipient] = (INF_SLEEP_TIME, 1.0)

    def get_sleep_times(self, game: Game) -> SleepTimes:
        assert self._sleep_times is not None, "Did you forget to call get_sleep_time first?"
        assert game_hash(game) == self._hash, "get_sleep_time was called on a different game state"
        return self._sleep_times


class ParlaiMessageHandler:
    def __init__(
        self,
        cfg: agents_cfgs.ParlaiDialogue,
        model_orders: Optional[BaseOrderWrapper] = None,
        base_strategy_model: Optional[BaseStrategyModelWrapper] = None,
    ):
        # load dialogue model
        assert cfg.model_dialogue is not None
        self.model_dialogue = load_dialogue_wrapper(cfg.model_dialogue)

        # load a sleep classifier model (required)
        self.model_sleep_classifier = load_sleep_classifier_wrapper(cfg.model_sleep_classifier)
        self.sleep_inf_threshold = cfg.sleep_inf_threshold
        self.sleep_inf_threshold_reply = cfg.sleep_inf_threshold_reply
        self.initiate_sleep_heuristic_every_phase = cfg.initiate_sleep_heuristic_every_phase
        self.use_initiate_sleep_heuristic_n_years = cfg.use_initiate_sleep_heuristic_n_years
        self.use_pseudoorders_initiate_sleep_heuristic = (
            cfg.use_pseudoorders_initiate_sleep_heuristic
        )
        self.use_last_phase_silence_except_coordination_heuristic = (
            cfg.use_last_phase_silence_except_coordination_heuristic
        )
        self.grounding_last_playable_year = cfg.grounding_last_playable_year
        self.dialogue_batch_size = cfg.dialogue_batch_size

        self.base_strategy_model = base_strategy_model
        self.block_initiation_if_pred_value_below = cfg.block_initiation_if_pred_value_below
        if self.block_initiation_if_pred_value_below > 0:
            assert (
                self.base_strategy_model is not None
            ), "Need a value model to use block_initiation_if_pred_value_below"

        # load a recipient classifier model if it exists
        self.model_recipient_classifier = (
            load_recipient_classifier_wrapper(cfg.model_recipient_classifier)
            if cfg.model_recipient_classifier.model_path
            else None
        )
        if self.model_recipient_classifier is not None:
            assert not self.model_recipient_classifier.expects_pseudo_orders(), "Deprecated"

        if self.model_dialogue.expects_recipient():
            # Must have a recipient classifier if the model expects a recipient
            assert (
                self.model_recipient_classifier is not None
                or self.model_sleep_classifier.is_sleepsix()
            )
            assert not (
                (self.model_recipient_classifier is not None)
                and self.model_sleep_classifier.is_sleepsix()
            ), "Do not load both sleepsix and recipient classifiers"

        self.model_draw_classifier = (
            load_draw_classifier_wrapper(cfg.model_draw_classifier)
            if cfg.model_draw_classifier.model_path
            else None
        )

        # DEPRECATED
        zshot_nonsense_classifier = None

        # maybe load discriminative nonsense classifier, and pass it to model_dialogue.message_editor
        ensemble_nonsense_classifier = load_ensemble_nonsense_classifier_wrapper(
            cfg.ensemble_nonsense_classifier
        )

        # instantiate message filter and editor
        self.message_filterer = MessageFiltering(
            filter_offensive_language=cfg.filter_offensive_dialogue,
            filter_phase_repeats=True,
            filter_consecutive_short=True,
            filter_excess_redacted=True,
            filter_any_redacted=True,
            filter_ampersands=True,
            filter_names=True,
            filter_urls_emails=True,
            filter_draw_discussion_when_missing_votes=True,
            filter_mutes=True,
            filter_grounding=cfg.should_filter_grounding,
            filter_insults=True,
            grounding_last_playable_year=cfg.grounding_last_playable_year,
            rating_threshold_first_message=cfg.rating_threshold_first_message,
            rating_threshold_other=cfg.rating_threshold_other,
            zshot_nonsense_classifier=zshot_nonsense_classifier,
            ensemble_nonsense_classifier=ensemble_nonsense_classifier,
            orders_model=model_orders
            if isinstance(model_orders, ParlAIAllOrderIndependentRolloutWrapper)
            else None,
            pseudo_orders_correspondence_threshold=cfg.pseudo_orders_correspondence_threshold,
            dialogue_model=self.model_dialogue
            if isinstance(self.model_dialogue, BaseDialogueWrapper)
            else None,
        )
        self.message_editor = MessageEditing(
            edit_newlines=True,
            # edit_names should NOT be True because the function of this parameter is to
            # redact out generated names and replace it with a redaction-like token.
            # At inference time when play real games, we should be filtering the whole
            # message with real life name (filter_names=True above), since of course
            # we don't want to actually be sending a redaction token in the output.
            edit_names=False,
            edit_weird_capitalization=True,
        )

        # resample dialogue on filter
        self.resample_dialogue_on_filter = cfg.resample_dialogue_on_filter

        # store the orders model in case we need it for
        # pseudo-orders correspondence filtering
        self.model_orders = model_orders

        # load pseudo orders model if it exists; else, use the provided orders model
        # Note: a "plausible pseudo-orders model" can (and should) be used here
        #
        # Note 2: We don't want to call load_pseudo_orders_wrapper, because that
        # loads a DevOnlyAnnotatedPseudoOrdersWrapper that's only used to annotate the training
        # set for dialogue models.
        if cfg.model_pseudo_orders.model_path:
            self.model_pseudo_orders_executor = load_wrapper_executor(
                cfg.model_pseudo_orders,
                load_order_wrapper,
                cfg.allow_multi_gpu,
                load_model_on_main=True,
            )
        else:
            assert model_orders is not None
            self.model_pseudo_orders_executor = wrap_parlai_model_to_executor(model_orders)
        self.model_pseudo_orders = cast(
            BaseOrderWrapper, self.model_pseudo_orders_executor.get_model()
        )

        if self.model_pseudo_orders.expects_recipient():
            assert (
                self.model_recipient_classifier is not None
                or self.model_sleep_classifier.is_sleepsix()
            ), "Need mechanism to choose recipient"

        self.reuse_pseudo_for_consecutive_messages = cfg.reuse_pseudo_for_consecutive_messages
        self.reuse_pseudo_for_phase = cfg.reuse_pseudo_for_phase

        # sleepsix model also acts as recipient classifier -- upon computing
        # sleep times, save the chosen recipient here alongside game hash and # messages
        self.cached_recipient: Tuple[Tuple[int, int], Power] = ((-1, -1), "NONE")

        # initial message prompting cfg
        self.initial_message_prompts_path = cfg.initial_message_prompts_path
        self.initial_message_prompts_count = cfg.initial_message_prompts_count
        self.initial_message_prompt_spacing_seconds = cfg.initial_message_prompt_spacing_seconds
        self.initial_message_prompts = (
            json.load(open(self.initial_message_prompts_path))
            if self.initial_message_prompts_path
            else None
        )

        self.binarize_sleep_times_in_5m_games = cfg.binarize_sleep_times_in_5m_games
        self.limit_consecutive_outbound_messages = cfg.limit_consecutive_outbound_messages

    def expects_pseudo_orders(self) -> bool:
        """Does any module expect pseudo orders?"""
        return self.model_dialogue.expects_pseudo_orders()

    def _check_valid_pseudo_orders(
        self, game: Game, pseudo_orders: PseudoOrders, possible_orders: Dict[Power, List[Order]]
    ) -> bool:
        """
        Check that pseudo orders are valid.

        Returns True/False corresponding to whether pseudo orders are valid
        """
        # Check that the current phase of actions is valid against all possible actions
        _, bad_orders = filter_orders_many_powers(
            pseudo_orders.first_joint_action(), possible_orders, subset_ok=True
        )

        # if all orders are valid
        if not all(len(orders) == 0 for orders in bad_orders.values()):
            return False

        if not self.rollout_pseudo_orders() or is_movement_phase(game.current_short_phase):
            return True

        # Specific checks for rollout pseudo orders on non-movement phases
        # First, check that the model indeed produced a rollout pseudo order
        valid_rollout = pseudo_orders.check_rollout(self.model_dialogue.expected_rollout_type())
        if not valid_rollout:
            logging.warning(
                f"Pseudo orders model did not produce valid rollout orders for phase {game.current_short_phase}:\n{pseudo_orders}"
            )
            return False

        # Next, check that future phase orders have a valid format
        # This is less strict than checking if the orders are valid
        for phase in pseudo_orders.phases():
            for _, orders in pseudo_orders[phase].items():
                for order in orders:
                    # Validate against v1 version of orders, which should match the formatting in the
                    # PseudoOrders object
                    match = re.fullmatch(InputValidator([], "", {}, 1).ORDER, order, re.VERBOSE)
                    if not match:
                        logging.warning(f"Pseudo order is mal-formatted: {order}, trying again")
                        return False

        return True

    def get_pseudo_orders_many_powers(
        self, game: Game, speaking_power: Power, recipient: Optional[Power] = None,
    ) -> Optional[PseudoOrders]:
        """
        Get pseudo orders for all powers from the perspective of a single power.
        """
        # we need to be able to predict orders for every power
        if self.model_pseudo_orders is None:
            return None

        possible_orders = get_all_possible_orders_by_power(game)

        assert MAX_ATTEMPTS_GEN_VALID > 0
        pseudo_orders = None
        # patterned after ParlaiOrderHandler._get_orders_single_power
        for i in range(MAX_ATTEMPTS_GEN_VALID):
            # generate some orders
            if not self.rollout_pseudo_orders():
                pred = self.model_pseudo_orders.produce_joint_action(game, speaking_power)
                pseudo_orders = PseudoOrders.from_joint_action(pred, game.current_short_phase)
            else:
                assert recipient is not None
                pred = self.model_pseudo_orders.produce_rollout_joint_action_bilateral(
                    game, speaking_power, recipient
                )
                pseudo_orders = PseudoOrders.from_rollout_joint_action(pred)

            valid_orders = self._check_valid_pseudo_orders(game, pseudo_orders, possible_orders)
            if valid_orders:
                if i > 0:
                    logging.warning(
                        f"ParlAI pseudo-orders model took {i + 1} attempts to produce a valid order"
                    )
                break

        logging.info(f"Pseudo order prediction for {speaking_power}: {pseudo_orders}")
        assert pseudo_orders is not None

        return pseudo_orders

    def get_recipient(
        self,
        game: Game,
        power: Power,
        timestamp: Optional[Timestamp] = None,
        sleepsix_cache: Optional[SleepSixTimesCache] = None,
    ) -> Optional[Power]:
        """
        Predict a recipient
        """
        if self.model_sleep_classifier.is_sleepsix():
            assert sleepsix_cache is not None
            if sleepsix_cache.should_recompute(game):
                logging.info(
                    f"Running sleep classifier from get_recipient, power={power} timestamp={timestamp}"
                )
                self.get_sleep_time(game, power, sleepsix_cache)
            return sleepsix_cache.get_recipient_sleep_time(game)[0]

        if self.model_recipient_classifier is None:
            return None

        assert timestamp is not None, "must specify timestamp for recipient classifier"
        return self.model_recipient_classifier.get_recipient(game, power, timestamp)

    def get_sleep_time(
        self,
        game: Game,
        power: Power,
        sleepsix_cache: Optional[SleepSixTimesCache] = None,
        recipient: Optional[Power] = None,
    ) -> Timestamp:
        assert self.has_sleep_classifier()
        assert not self.model_sleep_classifier.expects_pseudo_orders()

        if self.model_sleep_classifier.is_sleepsix():
            assert sleepsix_cache is not None

            if sleepsix_cache.should_recompute(game):
                # compute sleep times for each target
                targets = [p for p in game.get_alive_powers() if p != power]
                sleep_times = self.model_sleep_classifier.get_sleepsix_times(
                    game,
                    power,
                    targets,
                    inf_thresholds=[
                        self.sleep_inf_threshold_reply
                        if self.sleep_inf_threshold_reply > 0
                        and is_replying_to(game, power, target)
                        else self.sleep_inf_threshold
                        for target in targets
                    ],
                )

                # apply heuristics
                sleep_times = apply_needs_response_sleep_heuristic(game, power, sleep_times)
                if game.current_short_phase == "S1901M" or (
                    game.current_short_phase.endswith("M")
                    and (
                        self.initiate_sleep_heuristic_every_phase
                        or (
                            (int(game.current_short_phase[1:-1]) - 1901)
                            < self.use_initiate_sleep_heuristic_n_years
                        )
                    )
                ):
                    sleep_times = apply_initiate_sleep_heuristic(
                        game, power, sleep_times, restrict_to_powers=POWERS
                    )

                if self.binarize_sleep_times_in_5m_games:
                    sleep_times = apply_5m_binary_sleep_heuristic(game, power, sleep_times)
                if self.limit_consecutive_outbound_messages > 0:
                    sleep_times = apply_consecutive_outbound_messages_heuristic(
                        game, power, sleep_times, self.limit_consecutive_outbound_messages
                    )

                if (
                    self.block_initiation_if_pred_value_below > 0
                    and self.base_strategy_model is not None
                    and self.base_strategy_model.get_values(
                        game, has_press=True, agent_power=power
                    )[POWERS.index(power)]
                    < self.block_initiation_if_pred_value_below
                ):
                    logging.info("We are losing, calling apply_block_initiation_heuristic")
                    sleep_times = apply_block_initiation_heuristic(game, power, sleep_times)

                sleepsix_cache.set_sleep_times(game, sleep_times)

            if recipient is None:
                recipient, sleep_time = sleepsix_cache.get_recipient_sleep_time(game)
                logging.info(
                    f"sleepsix [{power}] choosing {recipient} ({sleep_time}) from {sleepsix_cache.get_sleep_times(game)}"
                )
            else:
                sleep_time, _ = sleepsix_cache.get_sleep_times(game)[recipient]
                logging.info(f"sleepsix [{power}] returning {sleep_time} for {recipient}")
            return sleep_time
        else:
            # legacy: separate sleep and recipient classifiers
            assert self.model_recipient_classifier is not None
            assert self.sleep_inf_threshold_reply == 0, "Only supported for sleepsix."
            return self.model_sleep_classifier.get_sleep_time(
                game, power, inf_threshold=self.sleep_inf_threshold
            )

    def has_sleep_classifier(self):
        return self.model_sleep_classifier is not None

    def _edit_messages(
        self,
        msg_dcts: List[MessageDict],
        game: Game,
        pseudo_orders: Optional[PseudoOrders],
        timings: Optional[TimingCtx] = None,
        skip_filtering: bool = False,
    ) -> List[MessageDict]:
        """
        Edit and or filter messages using the message filterer
        """
        filtered_msg_dcts = []
        for msg in msg_dcts:
            should_filter = False
            if not skip_filtering:
                logging.info(f"Running message filtering on message \"{msg['message']}\".")
                should_filter = self.message_filterer.should_filter_message(
                    msg,
                    list(game.messages.values()),
                    game,
                    pseudo_orders,
                    game_is_missing_draw_votes=False,
                    timings=timings,
                )
            if not should_filter:
                filtered_msg_dcts.append(self.message_editor.maybe_edit_message(msg))

        return filtered_msg_dcts

    def rollout_pseudo_orders(self) -> bool:
        """
        Return whether we need rollout pseudo orders for the dialogue model
        """
        return (
            self.model_dialogue.expects_pseudo_orders()
            and self.model_dialogue.expects_rollout_pseudo_orders()
        )

    def extended_rollout_pseudo_orders(self) -> bool:
        """
        Return whether we need extended rollout pseudo orders for the dialogue model
        """
        return (
            self.model_dialogue.expects_pseudo_orders()
            and self.model_dialogue.expects_extended_rollout_pseudo_orders()
        )

    def _maybe_produce_draw_message(
        self, game: Game, power: Power, timestamp: Timestamp
    ) -> Optional[MessageDict]:
        if self.model_draw_classifier is not None:
            draw_vote_status = self.model_draw_classifier.get_draw_vote_status(
                game, power, timestamp
            )
            if draw_vote_status == DrawVoteStatus.DRAW:
                logging.info(f"Power {power} voted for a draw")
                return build_draw_vote_message_dict(power, game.current_short_phase, timestamp)
            elif draw_vote_status == DrawVoteStatus.UNDRAW:
                logging.info(f"Power {power} UNvoted for a draw")
                return build_undraw_vote_message_dict(power, game.current_short_phase, timestamp)
        return None

    def _update_pseudo_orders(
        self, game: Game, power: Power, recipient: Power, pseudo_orders: Optional[PseudoOrders],
    ) -> Optional[PseudoOrders]:
        if self.model_dialogue.expects_pseudo_orders():
            if pseudo_orders is None:
                pseudo_orders = self.get_pseudo_orders_many_powers(
                    game, power, recipient=recipient
                )
            assert pseudo_orders is not None

            # sanity check
            for phase, joint_action in pseudo_orders.val.items():
                assert power in joint_action, f"{power} not in {phase}: {pseudo_orders}"
                if recipient is not None:
                    assert (
                        recipient in joint_action
                    ), f"{recipient} not in {phase}: {pseudo_orders}"

            logging.info(f"Pseudo orders for {power}: {pseudo_orders}")
            meta_annotations.add_pseudo_orders_next_msg(pseudo_orders)
            self.model_dialogue.update_pseudo_orders(
                game.current_short_phase, power, pseudo_orders
            )

            if (
                self.message_filterer.ensemble_nonsense_classifier
                and self.message_filterer.ensemble_nonsense_classifier.expects_pseudo_orders()
            ):
                self.message_filterer.ensemble_nonsense_classifier.update_pseudo_orders(
                    game.current_short_phase, power, pseudo_orders
                )

        return pseudo_orders

    def _produce_messages_maybe_with_annotations(
        self,
        game: Game,
        power: Power,
        timestamp: Timestamp,
        recipient: Power,
        pseudo_orders: Optional[PseudoOrders],
        n_messages: int = 1,
        timings: Optional[TimingCtx] = None,
        skip_filtering: bool = False,
    ) -> List[Tuple[MessageDict, Optional[MetaAnnotator]]]:
        if timings is None:
            timings = TimingCtx()

        base_meta_annotator = None
        if meta_annotations.has_annotator():
            base_meta_annotator = meta_annotations.pop_annotator()

        # timestamp_req: output message will be returned with this timestamp
        # timestamp_gen: used to condition dialogue model, may be modified to
        #                produce higher quality messages.
        timestamp_req = timestamp
        timestamp_gen = timestamp
        del timestamp

        # maybe apply initial message prompting -- timestamp_gen is the
        # timestamp used to condition the message generation. It may be
        # overridden by the initial message prompting, but timestamp_req will
        # be attached to the message upon return
        game_to_condition_on = game
        if self.initial_message_prompts:
            timings.start("initial_message_prompts")
            game_to_condition_on, timestamp_gen = maybe_apply_initial_message_prompting(
                game,
                power,
                recipient,
                timestamp_req,
                self.initial_message_prompts,
                self.initial_message_prompts_count,
                self.initial_message_prompt_spacing_seconds,
            )

        maybe_annotated_msg_dicts: List[Tuple[MessageDict, Optional[MetaAnnotator]]] = []
        filtered_annotators: List[MetaAnnotator] = []

        assert n_messages > 0
        if n_messages == 1:
            n_samples = self.resample_dialogue_on_filter + n_messages

            def _outbound_single_msg_dicts_generator():
                while True:
                    timings.start("produce_messages")
                    if recipient is not None and not self.model_dialogue.expects_recipient():
                        outbound_msg_dicts = self.model_dialogue.produce_messages_to_power_prefix(
                            game_to_condition_on,
                            power,
                            timestamp=timestamp_gen,
                            to_power=recipient,
                        )
                    else:
                        outbound_msg_dicts = self.model_dialogue.produce_messages(
                            game_to_condition_on,
                            power,
                            timestamp=timestamp_gen,
                            recipient=recipient,
                        )
                    timings.stop()
                    yield outbound_msg_dicts

            outbound_msg_dicts_generator = _outbound_single_msg_dicts_generator()
        else:
            n_samples = n_messages  # Don't retry if sampling multiple messages

            def _outbound_many_msg_dicts_generator():
                timings.start("produce_many_messages")
                assert isinstance(self.model_dialogue, BaseDialogueWrapper), type(
                    self.model_dialogue
                )
                if recipient is not None and not self.model_dialogue.expects_recipient():
                    outbound_msg_dicts_list = self.model_dialogue.produce_many_messages_to_power_prefix(
                        game_to_condition_on,
                        power,
                        timestamp=timestamp_gen,
                        num_preds=n_samples,
                        to_power=recipient,
                        batch_size=self.dialogue_batch_size,
                    )
                else:
                    outbound_msg_dicts_list = self.model_dialogue.produce_many_messages(
                        game_to_condition_on,
                        power,
                        timestamp=timestamp_gen,
                        num_preds=n_samples,
                        recipient=recipient,
                        batch_size=self.dialogue_batch_size,
                    )
                timings.stop()
                yield from map(lambda x: x[0], outbound_msg_dicts_list)

            outbound_msg_dicts_generator = _outbound_many_msg_dicts_generator()

        raise_if_should_stop(post_pseudoorders=True)

        for i in range(n_samples):
            assert not meta_annotations.has_annotator()
            if base_meta_annotator is not None:
                meta_annotations.push_annotator(
                    MetaAnnotator(game, base_meta_annotator.outpath, skip_start_tag=True)
                )

            outbound_msg_dicts = next(outbound_msg_dicts_generator)

            # Add outbound timestamp -- use timestamp_req even if a different
            # timestamp_gen was passed to the dialogue model
            msg_dicts = [with_time_sent(m, timestamp_req) for m in outbound_msg_dicts]

            # Now edit messages
            with timings.create_subcontext("filter_messages") as subtimings:
                msg_dicts = self._edit_messages(
                    msg_dicts, game, pseudo_orders, subtimings, skip_filtering=skip_filtering
                )

            assert len(msg_dicts) in [0, 1]
            if len(msg_dicts) == 0:
                n_remaining = n_messages + self.resample_dialogue_on_filter - i - 1
                logging.info(f"Message filtered. {n_remaining} attempts left")
                meta_annotations.after_message_generation_failed(bad_tags=[TOKEN_DETAILS_TAG])
                if base_meta_annotator is not None:
                    filtered_annotators.append(meta_annotations.pop_annotator())
            else:
                msg_dict = msg_dicts[0]
                msg_annotator = None
                if base_meta_annotator is not None:
                    msg_annotator = meta_annotations.pop_annotator()
                maybe_annotated_msg_dicts.append((msg_dict, msg_annotator))

            raise_if_should_stop(post_pseudoorders=True)

            if len(maybe_annotated_msg_dicts) == n_messages:
                break

        timings.stop()

        if base_meta_annotator is not None:
            meta_annotations.push_annotator(base_meta_annotator)
            for annotator in filtered_annotators:
                meta_annotations.append_annotator(annotator)
        return maybe_annotated_msg_dicts

    def generate_multiple_messages_with_annotations(
        self,
        game: Game,
        power: Power,
        timestamp: Timestamp,
        recipient: Power,
        pseudo_orders: Optional[PseudoOrders],
        n_messages: int,
        timings: Optional[TimingCtx] = None,
        skip_filtering: bool = False,
    ) -> List[Tuple[MessageDict, Optional[MetaAnnotator]]]:
        assert power in POWERS, power
        assert recipient in POWERS, recipient

        if timings is None:
            timings = TimingCtx()

        timings.start("init")

        # First, check if we potentially want to vote for a draw
        if self.model_draw_classifier is not None:
            with timings("get_draw_vote_status"):
                draw_msg_dict = self._maybe_produce_draw_message(game, power, timestamp)
                if draw_msg_dict is not None:
                    return [(draw_msg_dict, None)]

        assert (
            recipient in game.get_alive_powers()
        ), f"recipient={recipient}, alive_powers={game.get_alive_powers()}"

        timings.start("update_pseudo_orders")
        assert self.model_dialogue.expects_pseudo_orders()
        pseudo_orders = self._update_pseudo_orders(game, power, recipient, pseudo_orders)  # type: ignore

        # Shouldn't be here for S1901M first messages if we have initial_message_prompts
        assert not self.initial_message_prompts or not (
            game.current_short_phase == "S1901M" and (get_last_message_from(game, power) is None)
        ), (game.current_short_phase, get_last_message_from(game, power))

        return self._produce_messages_maybe_with_annotations(
            game,
            power,
            timestamp,
            recipient,
            pseudo_orders,
            n_messages,
            timings=timings,
            skip_filtering=skip_filtering,
        )

    def generate_message(
        self,
        game: Game,
        power: Power,
        timestamp: Timestamp,
        recipient: Power,
        pseudo_orders: Optional[PseudoOrders] = None,
        timings: Optional[TimingCtx] = None,
        skip_filtering: bool = False,
    ) -> Optional[MessageDict]:
        """Generate a single message, or None on failure.

        Expects pseudo orders as input for a pso-conditional dialogue model.

        Applies retry logic and message editing/filtering.
        """
        assert power in POWERS, power
        assert recipient in POWERS, recipient

        if timings is None:
            timings = TimingCtx()

        timings.start("init")

        # First, check if we potentially want to vote for a draw
        timings.start("get_draw_vote_status")
        draw_msg_dict = self._maybe_produce_draw_message(game, power, timestamp)
        if draw_msg_dict is not None:
            return draw_msg_dict

        assert (
            recipient in game.get_alive_powers()
        ), f"recipient={recipient}, alive_powers={game.get_alive_powers()}"

        timings.start("update_pseudo_orders")
        pseudo_orders = self._update_pseudo_orders(game, power, recipient, pseudo_orders)
        if self.model_dialogue.expects_pseudo_orders():
            assert pseudo_orders is not None

        maybe_annotated_msg_dicts = self._produce_messages_maybe_with_annotations(
            game,
            power,
            timestamp,
            recipient,
            pseudo_orders,
            timings=timings,
            skip_filtering=skip_filtering,
        )

        assert len(maybe_annotated_msg_dicts) in [0, 1]
        maybe_msg_dict = None
        if len(maybe_annotated_msg_dicts) == 1:
            maybe_msg_dict, maybe_annotator = maybe_annotated_msg_dicts[0]
            if maybe_annotator is not None:
                meta_annotations.append_annotator(maybe_annotator)

        timings.stop()

        self.message_filterer.report_statistics()  # log how many messages were filtered

        # Filtered messages/annotations will be saved even when no "final" message is available.
        self.message_filterer.report_filtering_annotations()

        return maybe_msg_dict


def apply_needs_response_sleep_heuristic(
    game: Game, power: Power, sleep_times: SleepTimes
) -> SleepTimes:
    """Constrain sleep times by "needs response" heuristic

    If a power has messaged us this phase and we have not yet messaged them
    back, ensure that we do so with a low sleep time, i.e. force at least one
    response per (phase, target)
    """
    phase_messages = game.messages.values()
    powers_who_messaged_us = {m["sender"] for m in phase_messages if m["recipient"] == power}
    powers_we_messaged = {m["recipient"] for m in phase_messages if m["sender"] == power}
    needs_response_powers = powers_who_messaged_us - powers_we_messaged
    if needs_response_powers:
        logging.info(
            f"Applying needs response sleep heuristic for {power}: {needs_response_powers}"
        )
    res = {}
    for p, t in sleep_times.items():
        if p in needs_response_powers:
            # Force response
            res[p] = (min(t[0], Timestamp.from_seconds(random.randint(0, 10 * 60))), t[1])
        else:
            # Keep original time
            res[p] = t

    return res


def apply_initiate_sleep_heuristic(
    game: Game, power: Power, sleep_times: SleepTimes, restrict_to_powers: List[Power],
) -> SleepTimes:
    """Constrain sleep times by forcing one message to each power"""
    phase_messages = game.messages.values()
    powers_we_messaged = {m["recipient"] for m in phase_messages if m["sender"] == power}
    needs_greet_powers = set(game.get_alive_powers()) - {power} - powers_we_messaged
    needs_greet_powers = needs_greet_powers.intersection(restrict_to_powers)
    if needs_greet_powers:
        logging.info(f"Applying initiate sleep heuristic for {power}: {needs_greet_powers}")

    # Iterating in sorted order uses original sleep times to tie-break powers
    # where the heuristic is applied.
    random_short_times = sorted(
        [Timestamp.from_seconds(random.randint(10 * 60, 30 * 60)) for _ in POWERS]
    )
    sleep_times_sorted = sorted(sleep_times.items(), key=lambda x: x[1])  # sort by time then prob
    return {
        p: ((min(t, random_short_time), prob) if p in needs_greet_powers else (t, prob))
        for (p, (t, prob)), random_short_time in zip(sleep_times_sorted, random_short_times)
    }


def apply_consecutive_outbound_messages_heuristic(
    game: Game, power: Power, sleep_times: SleepTimes, limit: int = 4
) -> SleepTimes:
    """Constrain sleep times by limiting consecutive outbound messages"""
    # count consecutive outbound messages to each recipient
    consecutive_outbound_messages = Counter()
    for _, msg in sorted(game.messages.items()):
        if msg["sender"] == power:
            consecutive_outbound_messages[msg["recipient"]] += 1
        else:
            consecutive_outbound_messages[msg["sender"]] = 0

    return {
        p: (t, prob) if consecutive_outbound_messages[p] < limit else (INF_SLEEP_TIME, 1.0)
        for p, (t, prob) in sleep_times.items()
    }


def apply_5m_binary_sleep_heuristic(
    game: Game,
    power: Power,
    sleep_times: SleepTimes,
    override_time: Timestamp = Timestamp.from_seconds(15),
) -> SleepTimes:
    """In 5m games, sleep times are 15 or inf

    override_time=0 could be used in webdip games (with the actual computation
    time providing space between messages) but a time of 15s is used by default
    to keep self-play games looking somewhat normal.

    1s offsets are added to preserve ordering.
    """
    if game.get_metadata("phase_minutes") != "5":
        return sleep_times

    sleep_times_sorted = sorted(sleep_times.items(), key=lambda x: x[1])  # sort by time then prob
    return {
        p: (
            (INF_SLEEP_TIME if t == INF_SLEEP_TIME else override_time + Timestamp.from_seconds(i)),
            prob,
        )
        for i, (p, (t, prob)) in enumerate(sleep_times_sorted)
    }


def apply_block_initiation_heuristic(
    game: Game, power: Power, sleep_times: SleepTimes,
):
    # who have we not exchanged messages with this phase?
    powers_to_block = set(sleep_times.keys())
    for m in game.messages.values():
        if m["recipient"] == "ALL":
            continue
        powers_to_block.discard(m["sender"])
        powers_to_block.discard(m["recipient"])
    logging.info(f"Blocking initiation to {powers_to_block}")
    return {
        p: (t, prob) if p not in powers_to_block else (INF_SLEEP_TIME, 1.0)
        for p, (t, prob) in sleep_times.items()
    }


def maybe_apply_initial_message_prompting(
    game: Game,
    power: Power,
    recipient: Power,
    timestamp: Timestamp,
    possible_prompts: List[Dict],
    prompts_count: int,
    prompts_spacing_seconds: int,
) -> Tuple[Game, Timestamp]:
    # only apply to first outbound message of the game
    if not (game.current_short_phase == "S1901M" and get_last_message_from(game, power) is None):
        return game, timestamp
    assert recipient is not None

    # don't mutate the original game object, return a copy
    game = Game(game)

    # choose prompts to `prompts_count` different recipients
    possible_prompts = [
        m for m in possible_prompts if m["sender"] == power and m["recipient"] != recipient
    ]
    possible_prompt_recipients = list({m["recipient"] for m in possible_prompts})
    random.shuffle(possible_prompt_recipients)
    prompt_recipients = random.sample(
        possible_prompt_recipients, min(prompts_count, len(possible_prompt_recipients)),
    )
    prompts = [
        random.choice([p for p in possible_prompts if p["recipient"] == prompt_recipient])
        for prompt_recipient in prompt_recipients
    ]

    # insert prompts
    next_timestamp = (
        Timestamp(0)
        if len(game.messages) == 0
        else list(game.messages.keys())[-1] + Timestamp.from_seconds(prompts_spacing_seconds)
    )
    for m in prompts:
        logging.info(
            f'Applying initial message prompt for [{timestamp}] {power} -> {recipient}: [{next_timestamp}] {m["sender"]} -> {m["recipient"]}: {m["message"]}'
        )
        game.add_message(m["sender"], m["recipient"], m["message"], next_timestamp)
        next_timestamp = next_timestamp + Timestamp.from_seconds(prompts_spacing_seconds)

    return game, next_timestamp


def pseudoorders_initiate_sleep_heuristics_should_trigger(
    game: Game, power: Power, recipient: Power, pseudo_orders: PseudoOrders,
) -> bool:
    """Returns true if pseudoorders sleep heuristics suggest
    that this is a phase where we should initiate a message"""

    # If anywhere in the pseudoorders we plan to support a player
    # then force a message to them.
    # We only look at the first phase pseudos because the later phases are not as
    # easy to check becuase the board state will have changed, plus sometimes the
    # pseudos show alliance switches and stuff and it makes more sense to talk
    # about that as it's going to happen.

    phase, joint_action = pseudo_orders.first_phase_and_joint_action()

    # Movement phases only
    if not phase.endswith("M"):
        return False

    # Initiation heuristic only triggers when we have not yet messaged the recipient
    if any(m["sender"] == power and m["recipient"] == recipient for m in game.messages.values()):
        return False

    # X-power supports and convoys should force a message
    if joint_action_contains_xpower_support_or_convoy(game, power, recipient, joint_action):
        logging.info(
            f"Forcing message to {recipient} due to pseudoorder giving xpower support/convoy {phase} {joint_action}"
        )
        return True

    # If making direct attacks against the same opponent as another player
    # then force a message to them.
    if power in joint_action and recipient in joint_action:
        for opponent in POWERS:
            if (
                opponent != power
                and opponent != recipient
                and action_has_any_direct_attack(game, joint_action[power], opponent)
                and action_has_any_direct_attack(game, joint_action[recipient], opponent)
            ):
                logging.info(
                    f"Forcing message to {recipient} due to pseudoorder common enemy {phase} {joint_action}"
                )
                return True

    return False


def joint_action_contains_xpower_support_or_convoy(
    game: Game, power: Power, recipient: Power, joint_action: JointAction
) -> bool:
    if power in joint_action:
        for order in joint_action[power]:
            if is_friendly_xpower_support_or_convoy(game, order, recipient):
                return True
    if recipient in joint_action:
        for order in joint_action[recipient]:
            if is_friendly_xpower_support_or_convoy(game, order, power):
                return True
    return False
