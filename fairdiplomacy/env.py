#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import json
import logging
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart
import numpy as np
import os
import random
import torch
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.base_search_agent import BaseSearchAgent
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.player import Player
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils import game_scoring
from fairdiplomacy.utils.atomicish_file import atomicish_open_for_writing
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.typedefs import get_last_message
from fairdiplomacy.utils.yearprob import get_prob_of_latest_year_leq
import fairdiplomacy.variance_reduction
from fairdiplomacy.typedefs import (
    JointAction,
    MessageDict,
    Phase,
    Policy,
    Power,
    Timestamp,
)
from fairdiplomacy.viz.meta_annotations import api as meta_annotations

PYDIPCC_MAX_YEAR = 1935
STATE_SUFFIX = ".state.bin"
MIN_INTERVAL = Timestamp.from_centis(1)


class BasePolicyProfile(ABC):
    """
    Abstract base class for a policy profile.
    NOT intended for use beyond one game, since implementations also generally track agent state.
    """

    @abstractmethod
    def get_player(self, power: Power) -> Player:
        pass

    def state_dict(self) -> Any:
        return {p: self.get_player(p).state_dict() for p in POWERS}

    def load_state_dict(self, state_dict: Any):
        for p in POWERS:
            self.get_player(p).load_state_dict(state_dict[p])

    @abstractmethod
    def get_all_power_orders(self, game: Game) -> JointAction:
        """
        Generate orders for all powers given a game object.

        Child classes must implement this function.
        """
        pass

    def generate_messages_single_power(
        self, game: Game, power: Power, timestamp: Timestamp
    ) -> List[MessageDict]:
        """Return a list of message dicts expected by the game environment.

        Calls agent.generate_message and formats into a list of message dicts.

        Arguments:
        - game: a fairdiplomacy.Game object
        - power: (string) power to generate messages for

        Returns: List[Dict]
            A list of message dicts for that power.
        """
        player = self.get_player(power)
        game_view = game_from_view_of(game, power)
        msg = player.generate_message(game_view, timestamp)
        if msg is None:
            # All messages were filtered out. Dropping all annotations.
            meta_annotations.after_message_generation_failed()
        return [msg] if msg else []

    def agents_can_sleep(self) -> bool:
        """
        Returns whether agents in the policy profile can sleep; child classes should override
        """
        can_sleep = [self.get_player(pwr).can_sleep() for pwr in POWERS]
        assert len(set(can_sleep)) == 1, "All players must sleep, or none"
        return list(can_sleep)[0]

    def get_sleep_time(self, game: Game, power: Power) -> Timestamp:
        """
        Used in full press games to predict time before next message is sent
        """
        if not self.agents_can_sleep():
            raise RuntimeError("Agents cannot sleep")

        possible_recipients = set(game.get_alive_powers()) - {power}
        player = self.get_player(power)
        return min(
            [
                player.get_sleep_time(game_from_view_of(game, power), recipient)
                for recipient in possible_recipients
            ]
        )


class OneSixPolicyProfile(BasePolicyProfile):
    """A combination of independent agents to predict all power moves."""

    def __init__(
        self,
        agent_one: BaseAgent,
        agent_six: BaseAgent,
        agent_one_power: Power,
        game: Game,  # unused, but present to discourage use of this object beyond one game
        share_strategy=False,
    ):
        self._agent_one_power = agent_one_power
        self._six_powers = [p for p in POWERS if p != agent_one_power]
        self._agent_one = agent_one
        self._agent_six = agent_six
        self._players = {
            p: Player(agent_one if p == agent_one_power else agent_six, p) for p in POWERS
        }
        self.share_strategy = share_strategy

    def get_player(self, power: Power) -> Player[BaseAgent]:
        return self._players[power]

    def get_all_power_orders(self, game) -> JointAction:
        logging.debug("Starting turn {}".format(game.phase))
        orders = {}
        log_getting_agent_orders_for([self._agent_one_power])
        orders[self._agent_one_power] = self._players[self._agent_one_power].get_orders(
            game_from_view_of(game, self._agent_one_power)
        )
        if self.share_strategy and self._agent_six.can_share_strategy():
            assert not getattr(
                self._agent_six, "use_final_iter", False
            ), "Unsafe: share_strategy + use_final_iter"
            log_getting_agent_orders_for(self._six_powers)
            orders.update(self._agent_six.get_orders_many_powers(game, self._six_powers))
        else:
            for pwr in self._six_powers:
                log_getting_agent_orders_for([pwr])
                orders[pwr] = self._players[pwr].get_orders(game_from_view_of(game, pwr))
        return orders

    def get_all_power_orders_and_agent_one_policy(self, game) -> Tuple[JointAction, Power, Policy]:
        logging.debug("Starting turn {}".format(game.phase))
        assert isinstance(self._agent_one, BaseSearchAgent)
        log_getting_agent_orders_for([self._agent_one_power])
        policies = (
            self.get_player(self._agent_one_power)
            .run_search(game, allow_early_exit=True)
            .get_agent_policy()
        )
        assert not getattr(self._agent_one, "use_final_iter", False)
        orders = {}
        orders[self._agent_one_power] = sample_p_dict(policies[self._agent_one_power])
        if self.share_strategy and self._agent_six.can_share_strategy():
            log_getting_agent_orders_for(self._six_powers)
            orders.update(self._agent_six.get_orders_many_powers(game, self._six_powers))
        else:
            for pwr in self._six_powers:
                log_getting_agent_orders_for([pwr])
                orders[pwr] = self.get_player(pwr).get_orders(game_from_view_of(game, pwr))

        return orders, self._agent_one_power, policies[self._agent_one_power]


class PopulationPolicyProfile(BasePolicyProfile):
    def __init__(
        self,
        power_agent_dict: Dict[Power, BaseAgent],
        game: Game,  # unused, but present to discourage use of this object beyond one game
    ):
        self._players = {power: Player(agent, power) for power, agent in power_agent_dict.items()}

    def get_player(self, power: Power) -> Player[BaseAgent]:
        return self._players[power]

    def get_all_power_orders(self, game) -> JointAction:
        logging.debug("Starting turn {}".format(game.phase))
        orders = {}
        for pwr, agent in self._players.items():
            log_getting_agent_orders_for([pwr])
            orders[pwr] = agent.get_orders(game_from_view_of(game, pwr))

        return orders


class SharedPolicyProfile(BasePolicyProfile):
    """Single agent that predicts orders for all powers."""

    def __init__(
        self,
        agent: BaseAgent,
        game: Game,  # unused, but present to discourage use of this object beyond one game
        share_strategy=False,
    ):
        self._agent = agent
        self._players = {p: Player(agent, p) for p in POWERS}
        self.share_strategy = share_strategy

    def get_player(self, power: Power) -> Player[BaseAgent]:
        return self._players[power]

    def get_all_power_orders(self, game) -> JointAction:
        if self.share_strategy and self._agent.can_share_strategy():
            assert not getattr(
                self._agent, "use_final_iter", False
            ), "Unsafe: share_strategy + use_final_iter"
            log_getting_agent_orders_for(POWERS)
            return self._agent.get_orders_many_powers(game, POWERS)
        else:
            orders = {}
            for pwr in POWERS:
                log_getting_agent_orders_for([pwr])
                orders[pwr] = self.get_player(pwr).get_orders(game_from_view_of(game, pwr))
            return orders


class Env:
    def __init__(
        self,
        policy_profile: BasePolicyProfile,
        *,
        seed=0,
        cf_agent=None,
        max_year: int = PYDIPCC_MAX_YEAR,
        game: Game,
        max_msg_iters=-1,
        capture_logs=False,
        time_per_phase: int = 8640000,  # centiseconds
        variance_reduction_model: Optional[BaseStrategyModelWrapper] = None,
        stop_when_power_is_dead: Optional[Power] = None,
        year_spring_prob_of_ending: Optional[Dict[int, float]] = None,
    ):
        self.game = game

        # set phase length in *minutes*; time_per_phase is in *centiseconds*
        phase_minutes = (time_per_phase // 100) // 60
        self.game.set_metadata("phase_minutes", str(phase_minutes))

        # set random seeds
        random.seed(seed)
        np.random.seed(seed)  # type:ignore
        torch.manual_seed(seed)

        self.policy_profile = policy_profile
        self.cf_players = {p: Player(cf_agent, p) for p in POWERS} if cf_agent else {}
        assert (
            max_year <= PYDIPCC_MAX_YEAR
        ), f"pydipcc doesn't allow to go beyond {PYDIPCC_MAX_YEAR}"
        self.max_year = max_year
        self.capture_logs = capture_logs

        self.variance_reduction_model = variance_reduction_model
        self.variance_reduction_offsets_by_phase = {}
        self.policies_by_phase = {}
        self.stop_when_power_is_dead = stop_when_power_is_dead
        self.year_spring_prob_of_ending = year_spring_prob_of_ending

        # Get last timestamp if we initialize with an existing game
        last_known_message = get_last_message(game)
        last_known_timestamp = (
            last_known_message[MessageObjectPart.TIME_SENT]
            if last_known_message is not None
            else None
        )

        self.message_runner = TimeBasedMessageRunner(
            self.policy_profile,
            run_limit=Timestamp.from_centis(time_per_phase),
            max_msg_iters=max_msg_iters,
            last_known_timestamp=last_known_timestamp,
        )

    def process_turn(self, timeout=10):
        log_capture_handler = None
        log_capture_io = None
        if self.capture_logs:
            log_capture_io = StringIO()
            log_capture_handler = logging.StreamHandler(log_capture_io)
            log_capture_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(log_capture_handler)
        else:
            log_capture_io = None
            log_capture_handler = None

        # run the messaging portion
        if self.policy_profile.agents_can_sleep():
            logging.info("Starting messaging for turn {}".format(self.game.phase))
            self.message_runner.run(self.game)

        # run order prediction
        logging.info("Starting order prediction for turn {}".format(self.game.phase))

        phase = self.game.current_short_phase
        if self.variance_reduction_model is not None:
            assert isinstance(self.policy_profile, OneSixPolicyProfile)
            (
                power_orders,
                agent_one_power,
                agent_one_policy,
            ) = self.policy_profile.get_all_power_orders_and_agent_one_policy(self.game)

            variance_reduction_offset = fairdiplomacy.variance_reduction.compute_variance_reduction_offsets(
                self.variance_reduction_model,
                self.game,
                agent_one_power,
                agent_one_policy,
                power_orders,
            )
            self.variance_reduction_offsets_by_phase[phase] = variance_reduction_offset

            self.policies_by_phase[phase] = list(agent_one_policy.items())
        else:
            power_orders = self.policy_profile.get_all_power_orders(self.game)

        logging.info("Finished order prediction for turn {}".format(self.game.phase))

        for power, orders in power_orders.items():
            if not self.game.get_orderable_locations().get(power):
                logging.debug(f"Skipping orders for {power}")
                continue
            logging.info("Set orders {} {} {}".format(phase, power, orders))
            if self.cf_players:
                cf_orders = self.cf_players[power].get_orders(self.game)
                logging.debug("CF  orders {} {} {}".format(phase, power, cf_orders))
            self.game.set_orders(power, orders)

        if log_capture_handler is not None:
            assert log_capture_io is not None
            logging.getLogger().removeHandler(log_capture_handler)
            log_capture_handler.close()
            self.game.add_log(log_capture_io.getvalue())

        self.game.process()
        meta_annotations.after_new_phase(self.game)

    def state_dict(self) -> Dict:
        return {
            "version": 1,
            "turn_id": self.turn_id,
            "seeds": {
                "random": random.getstate(),
                "torch": torch.get_rng_state(),
                "np": np.random.get_state(),  # type:ignore
            },
            "game": self.game.to_json(),
            "policy_profile": self.policy_profile.state_dict(),
            "cf_players": {power: v.state_dict() for power, v in self.cf_players.items()},
            "message_runner": self.message_runner.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        random.setstate(state["seeds"]["random"])
        np.random.set_state(state["seeds"]["np"])  # type:ignore
        torch.set_rng_state(state["seeds"]["torch"])
        self.game.from_json_inplace(state["game"])
        self.policy_profile.load_state_dict(state["policy_profile"])
        self.cf_players = state["cf_players"]
        for power, v in self.cf_players:
            v.load_state_dict(state["cf_players"][power])
        self.message_runner.load_state_dict(state["message_runner"])
        self.turn_id = state["turn_id"]

    def process_all_turns(
        self, max_turns: Optional[int] = 0, partial_out_name: Optional[str] = None,
    ) -> Dict[Power, int]:
        """Process all turns until game is over

        Returns a dict mapping power -> supply count
        """
        self.turn_id = 0
        if partial_out_name and os.path.exists(partial_out_name + STATE_SUFFIX):
            logging.warning(
                "Restoring eval process from checkpoint: %s", partial_out_name + STATE_SUFFIX
            )
            self.load_state_dict(torch.load(partial_out_name + STATE_SUFFIX))
            logging.warning("New phase: %s", self.game.get_current_phase())
        while not self.game.is_game_done:
            if max_turns and self.turn_id >= max_turns:
                break
            if (
                self.stop_when_power_is_dead is not None
                and self.stop_when_power_is_dead not in self.game.get_alive_powers()
            ):
                logging.info("Early stopping as agent %s is dead", self.stop_when_power_is_dead)
                break
            _, year, _ = self.game.phase.split()
            if int(year) > self.max_year:
                logging.info("Early stopping at %s due to reaching max year", year)
                break
            if (
                self.year_spring_prob_of_ending is not None
                and self.game.current_short_phase.startswith("S")
                and self.game.current_short_phase.endswith("M")
            ):
                if random.random() < get_prob_of_latest_year_leq(
                    self.year_spring_prob_of_ending, int(year)
                ):
                    logging.info(
                        "Early stopping at %s due to year_spring_prob_of_ending=%s",
                        year,
                        self.year_spring_prob_of_ending,
                    )
                    break

            self.process_turn()
            self.turn_id += 1

            if partial_out_name:
                # Save everything only to one file, as well as using the tmpfile and rename
                # trick, so that if failure happens exactly at the wrong moment, we still don't
                # end up with corrupt partially-written files.
                torch.save(self.state_dict(), partial_out_name + STATE_SUFFIX + ".tmp")
                meta_annotations.commit_annotations(self.game)
                os.rename(
                    partial_out_name + STATE_SUFFIX + ".tmp", partial_out_name + STATE_SUFFIX
                )
                # Also save the game by itself, just to help visualization/debugging
                with open(partial_out_name, "w") as stream:
                    stream.write(self.game.to_json())

        logging.info(f"Executed {self.turn_id} / {max_turns} turns.")
        if partial_out_name:
            if os.path.exists(partial_out_name):
                os.remove(partial_out_name)
            if os.path.exists(partial_out_name + STATE_SUFFIX):
                os.remove(partial_out_name + STATE_SUFFIX)
            if os.path.exists(partial_out_name + STATE_SUFFIX + ".tmp"):
                os.remove(partial_out_name + STATE_SUFFIX + ".tmp")

        return {k: len(v) for k, v in self.game.get_state()["centers"].items()}

    def save(self, output_path):
        game_json = self.game.to_json()

        # Save critical metadata info first
        info_path = output_path.replace(".json", ".info")
        logging.info("Saving scores and variance reduction offsets to {}".format(info_path))
        info = {}
        info["game_scores"] = {
            # _asdict() because named tuples don't jsonify well
            power: game_scoring.compute_game_scores(
                POWERS.index(power), json.loads(game_json)
            )._asdict()
            for power in POWERS
        }
        info["variance_reduction_offsets_by_phase"] = self.variance_reduction_offsets_by_phase
        # go ahead and record the full policy of agent one, in case we want to do later analysis
        # that wants to do variance reduction differently
        info["policies_by_phase"] = self.policies_by_phase
        with atomicish_open_for_writing(info_path, binary=False) as f:
            f.write(json.dumps(info))

        # save JSON next - so that this file's existence can act as a signifier that
        # everything is complete.
        logging.info("Saving game to {}".format(output_path))
        with atomicish_open_for_writing(output_path, binary=False) as stream:
            stream.write(game_json)


def get_alive_powers(game: Game) -> List[Power]:
    return game.get_alive_powers()


def add_messages_to_game_object(game: Game, messages: List[MessageDict]) -> None:
    # add all of the messages sent in a round to the game object, excluding
    # messages to dead powers
    alive_powers = get_alive_powers(game)
    assert alive_powers[0] == alive_powers[0].upper(), "Bad power name formatting!"

    for message in messages:
        game.add_message(
            message["sender"], message["recipient"], message["message"], message["time_sent"]
        )
        logging.info(
            f"({int(message['time_sent'])}) {message['sender']} -> {message['recipient']}: {message['message']}"
        )
        meta_annotations.after_message_add(list(game.messages.values())[-1])


def _get_affected_powers(msg_dicts: List[MessageDict]):
    affected = []
    for msg in msg_dicts:
        if msg["recipient"] == "ALL":
            return POWERS
        elif msg["recipient"] not in affected:
            affected.append(msg["recipient"])
    return affected


class AbstractMessageRunner(ABC):
    def __init__(
        self,
        policy_profile: BasePolicyProfile,
        run_limit: Timestamp,
        last_known_timestamp: Optional[Timestamp] = None,
    ):
        self.policy_profile = policy_profile
        self.run_limit = run_limit
        self.total_time = (
            MIN_INTERVAL if last_known_timestamp is None else last_known_timestamp
        )  # time for the entire game
        self.reset()

    def reset(self):
        """
        Reset the messaging loop. Called after every phase.
        """
        self.game: Optional[Game] = None
        self.curr_phase: Optional[Phase] = None
        self.curr_speaker: Optional[Power] = None

    @abstractmethod
    def round_over(self) -> bool:
        """
        Must return True if the round of message is over for a given phase,
        and False otherwise.

        Child classes must implement
        """
        pass

    @abstractmethod
    def message_iteration(self):
        """
        Runs through a single message iteration in a round.

        Child classes must implement
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a representation of any long-term state of this message runner that
        can be loaded back in with load_state_dict. Values in the dict should be
        torch checkpoint-serializable.
        Only game-long persistent state needs to be saved. Ephemeral variables
        that are only needed during a round that are initialized in initialize_round()
        and whose values don't matter beyond cleanup() or reset() do not need to be saved.

        Child classes may override to save their own values.
        """
        return dict(total_time=self.total_time)

    def load_state_dict(self, state: Dict[str, Any]):
        """
        Set the state of this message runner to a state produced by state_dict().

        Child classes may override to load their own values.
        """
        self.total_time = state["total_time"]

    def initialize_round(self, game):
        """
        Do any required setup for iterating through the messaging loop for a given phase

        Child classes may override to provide appropriate round initialization.
        """
        self.game = game
        self.curr_phase = self.game.current_short_phase

    def cleanup(self):
        """
        Optional: perform any required cleanup after a round of messaging, like logging.

        Left for child classes to override
        """
        pass

    def get_message_from_speaker(self, curr_speaker, timestamp: Timestamp) -> List[MessageDict]:
        """
        Helper method to get list of messages from a ParlAI agent
        """
        assert self.game is not None
        return self.policy_profile.generate_messages_single_power(
            self.game, curr_speaker, timestamp
        )

    def run(self, game):
        """
        Run through the messaging loop to get all messages for a phase.
        """
        self.initialize_round(game)
        while not self.round_over():
            self.message_iteration()

        self.cleanup()
        self.reset()


class TimeBasedMessageRunner(AbstractMessageRunner):
    def __init__(
        self,
        policy_profile,
        run_limit: Timestamp,
        max_msg_iters: int,
        last_known_timestamp: Optional[Timestamp] = None,
    ):
        super().__init__(policy_profile, run_limit, last_known_timestamp)
        logging.info(f"Using TimeBasedMessageRunner with a max time of {self.run_limit}cs")
        self.max_msg_iters = max_msg_iters

    def _log_sleep_times(self):
        logging.info(
            f"Current sleep times: {self.power_to_sleep_time_remaining} "
            f"elapsed_time {self.phase_time_so_far}/{self.run_limit}"
        )
        # WARNING: Don't change this log line
        # We use this for processing pseudo-orders
        logging.info(f"Total time elapsed for Message Runner: {self.total_time}")

    def reset(self):
        super().reset()
        self.phase_time_so_far = Timestamp.from_centis(0)
        self.power_to_sleep_time_remaining = {}
        self.num_phase_messages = 0

    def round_over(self) -> bool:
        """
        A round is over when all players have decided to
        sleep for longer than the allowed time limit.
        """
        assert self.run_limit is not None
        if self.max_msg_iters >= 0 and self.num_phase_messages >= self.max_msg_iters:
            return True
        return not any(
            [
                self.phase_time_so_far + t < self.run_limit
                for t in self.power_to_sleep_time_remaining.values()
            ]
        )

    def initialize_round(self, game):
        """
        Override from parent class to get initial sleep times
        for each power.
        """
        super().initialize_round(game)
        assert self.game is not None
        self.power_to_sleep_time_remaining = {
            power: self.policy_profile.get_sleep_time(self.game, power)
            for power in get_alive_powers(game)
        }
        self._log_sleep_times()

    def _update_sleep_times(
        self, curr_speaker: Power, slept_for: Timestamp, msg_dicts: List[MessageDict]
    ):
        """
        Sleep times are updated for each power after every message iteration.
        """
        affected_powers = _get_affected_powers(msg_dicts)
        assert self.game is not None
        for power in self.power_to_sleep_time_remaining:
            if power == curr_speaker or power in affected_powers:
                # Recalculated sleep time for interrupted powers
                self.power_to_sleep_time_remaining[power] = self.policy_profile.get_sleep_time(
                    self.game, power
                )
            else:
                # Update sleep time for other powers
                self.power_to_sleep_time_remaining[power] = (
                    self.power_to_sleep_time_remaining[power] - slept_for
                )

        self._log_sleep_times()

    def message_iteration(self):
        """
        We choose a player to speak who has the shortest sleep time.

        After they speak, we update the existing sleep times for all players
        """
        assert self.game is not None, "Not initialized"

        # Find the shortest sleep time
        slept_for: Timestamp = min(self.power_to_sleep_time_remaining.values())
        # Avoid collisions
        offset: Timestamp = MIN_INTERVAL if slept_for < MIN_INTERVAL else Timestamp.from_centis(0)

        # If multiple powers have the same sleep time, choose a random one
        curr_speaker = random.choice(
            [
                p
                for p in self.power_to_sleep_time_remaining
                if self.power_to_sleep_time_remaining[p] == slept_for
            ]
        )
        self.phase_time_so_far += slept_for + offset
        self.total_time += slept_for + offset

        # Get dialogue messages from the current speaker
        msg_dicts = self.get_message_from_speaker(curr_speaker, self.total_time)

        assert (
            len(msg_dicts) <= 1
        ), "Can only produce one message at a time in sleep-based messaging loop"

        # Add messages to the game object
        assert self.game is not None
        add_messages_to_game_object(self.game, msg_dicts)
        self.num_phase_messages += 1
        # Update sleep times
        self._update_sleep_times(curr_speaker, slept_for, msg_dicts)


def log_getting_agent_orders_for(powers: List[Power]):
    logging.info(f"GETTING AGENT ORDERS FOR: {powers}")
