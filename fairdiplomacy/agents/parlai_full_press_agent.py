#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Sequence, Optional, List

from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import AgentState, BaseAgent
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power, Timestamp, MessageDict
from fairdiplomacy.utils.typedefs import get_last_message
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME
from parlai_diplomacy.wrappers.factory import load_order_wrapper

from .parlai_message_handler import (
    ParlaiMessageHandler,
    ParlaiMessagePseudoOrdersCache,
    SleepSixTimesCache,
)
from .parlai_order_handler import ParlaiOrderHandler


class ParlaiAgentState(AgentState):
    def __init__(self):
        self.pseudo_orders_cache = ParlaiMessagePseudoOrdersCache()
        self.sleepsix_cache = SleepSixTimesCache()


class ParlaiFullPressAgent(BaseAgent):
    def __init__(self, cfg: agents_cfgs.ParlaiFullPressAgent):
        # Required: Orders Model
        self.model_orders = load_order_wrapper(cfg.model_orders)
        self.order_handler = ParlaiOrderHandler(self.model_orders)

        # Required: Message Handler
        assert cfg.dialogue is not None
        self.message_handler = ParlaiMessageHandler(cfg.dialogue, model_orders=self.model_orders)

    def initialize_state(self, power: Power) -> AgentState:
        return ParlaiAgentState()

    def get_orders(self, game: Game, power: Power, state: AgentState):
        return self.order_handler.get_orders(game, power)

    def can_sleep(self) -> bool:
        return self.message_handler is not None and self.message_handler.has_sleep_classifier()

    def get_sleep_time(
        self, game: Game, power: Power, state: AgentState, recipient: Optional[Power] = None,
    ) -> Timestamp:
        if not self.can_sleep():
            raise RuntimeError("This agent doesn't know how to sleep.")
        assert self.message_handler is not None
        assert isinstance(state, ParlaiAgentState)

        return self.message_handler.get_sleep_time(
            game, power, sleepsix_cache=state.sleepsix_cache, recipient=recipient,
        )

    def get_pseudo_orders(
        self, game: Game, power: Power, state: ParlaiAgentState, recipient: Optional[Power] = None,
    ) -> Optional[PseudoOrders]:
        # Get pseudo orders
        pseudo_orders = None
        cache = getattr(state, "pseudo_orders_cache", None)
        if cache is not None:
            pseudo_orders = cache.maybe_get(
                game,
                power,
                self.message_handler.reuse_pseudo_for_consecutive_messages,
                self.message_handler.reuse_pseudo_for_phase,
                recipient=recipient,
            )
        if pseudo_orders is None:
            pseudo_orders = self.message_handler.get_pseudo_orders_many_powers(
                game, power, recipient=recipient
            )
        if cache is not None and pseudo_orders is not None:
            cache.set(game, pseudo_orders, recipient=recipient)
        return pseudo_orders

    def generate_message(
        self,
        game: Game,
        power: Power,
        timestamp: Optional[Timestamp],
        state: ParlaiAgentState,
        recipient: Optional[Power] = None,
        pseudo_orders: Optional[PseudoOrders] = None,
    ) -> Optional[MessageDict]:
        # Fancy message re-generation only works with sleepsix code, not legacy code
        if not timestamp:
            sleep_time = self.get_sleep_time(game, power, state, recipient=recipient,)
            last_msg_dct = get_last_message(game)
            last_message_ts = (
                last_msg_dct["time_sent"] if last_msg_dct else Timestamp.from_seconds(0)
            )

            # To keep model in-distribution when force-sending messages with inf sleep time, condition on sleep time of 1 hour instead
            if sleep_time >= INF_SLEEP_TIME:
                if game.get_metadata("phase_minutes") == "5":
                    sleep_time = Timestamp.from_seconds(15)
                else:
                    sleep_time = Timestamp.from_seconds(60 * 60)

            timestamp = last_message_ts + sleep_time

        if recipient is None:
            recipient = self.message_handler.get_recipient(
                game, power, timestamp, state.sleepsix_cache
            )
        assert recipient is not None

        maybe_msg_dict = self.message_handler.generate_message(
            game, power, timestamp, recipient, pseudo_orders
        )

        if self.message_handler.model_sleep_classifier.is_sleepsix():
            if maybe_msg_dict is None:
                logging.info(f"Blocking messages to power: {recipient}")
                state.sleepsix_cache.block_messages_to_power(game, recipient)

        return maybe_msg_dict
