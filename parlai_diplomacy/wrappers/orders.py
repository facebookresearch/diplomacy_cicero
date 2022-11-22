#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Module to wrap ParlAI agent to produce orders given a game object.
"""
from abc import ABC
import re
from fairdiplomacy.timestamp import Timestamp
import logging
from typing import Dict, List, Tuple, Optional


from parlai_diplomacy.utils.game2seq.factory import get_output_type
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.utils.misc as misc
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper, RolloutType
from parlai_diplomacy.utils.game2seq.format_helpers.orders import (
    extract_rollout_action_for_power,
    is_movement_phase,
)

from fairdiplomacy import pydipcc
from fairdiplomacy.game import POWERS, sort_phase_key
from fairdiplomacy.typedefs import (
    Action,
    JointAction,
    JointPolicy,
    Phase,
    Power,
    RolloutAction,
    RolloutJointAction,
)

from fairdiplomacy.utils.slack import GLOBAL_SLACK_EXCEPTION_SWALLOWER

load.register_all_agents()


def assert_is_dialogue_phase(game: pydipcc.Game):
    cur_phase = game.current_short_phase
    last_dialogue_phase = game.get_metadata("last_dialogue_phase")
    assert last_dialogue_phase in ("", cur_phase), f"{cur_phase} != {last_dialogue_phase}"


def _find_scoring_for_actions(
    actions: List[Action], candidate_seqs: List[str], candidate_seqs_and_logps: Dict[str, float]
) -> List[Tuple[Action, float]]:
    ret = []
    assert len(candidate_seqs) == len(actions)
    for action, seq in zip(actions, candidate_seqs):
        if seq not in candidate_seqs_and_logps:
            logging.warning(
                f"WARNING: When rescoring, action {action} {seq} WAS NOT FOUND in the returned candidate_seqs_and_logps {candidate_seqs_and_logps}. Pretending it has roughly 0 probability."
            )
            # -500 log probability is basically like 0 probability, but exp() of it still produces
            # a positive double-precision float and so avoids divide by zero if we try to normalize
            # a policy that contains this and nothing else
            ret.append((action, -500.0))
        else:
            ret.append((action, candidate_seqs_and_logps[seq]))
    return ret


def coast_unqualify_supports(a: Action) -> Action:
    """Remove coast-qualified supports, which are inconsistent between parlai and base_strategy_model.

    """
    return tuple(re.sub(r"(S . ...)/.C$", r"\1", x) for x in a)


class BaseOrderWrapper(BaseWrapper, ABC):
    """
    Base class for wrappers for models that predicts orders.
    """

    OPT_OVERRIDE = {
        "interactive_mode": True,
        "skip_generation": False,
        "beam_length_penalty": 0,  # For orders models, we turn off length penalties
    }

    def expects_recipient(self) -> bool:
        """
        Whether the model expects a message recipient in order to produce orders.

        This is useful, e.g., for producing pseudo orders targeted towards a specific recipient.
        """
        return False

    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Optional[Power] = None,
        target_power: Power = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Given a game object, return a dictionary of formatted input sequences for each
        power to feed directly to the model

        Args:
            game: dipcc game object
            view_of_power: power with the game view
            target_power: (Optional) target recipient; this is useful for predicting pseudo orders
            timestamp: (Optional) unused in the orders wrappers -- current timestamp

        Returns:
            (str) Input sequence for predicting orders from a view of power
        """
        seqs = self.formatter.change_format(game, self.input_format_str, self.metadata)
        seq = misc.last_dict_value(seqs)[view_of_power]["input"]
        return seq

    def format_output_seq_action(self, output_seq: str, power: Power, phase: Phase) -> Action:
        """
        Given an output sequence, return the Action for a given power

        Args:
            output_seq: (str) output string produced by the model
            power: power taking the action
            phase: current phase; it is necessary to specify this, e.g., for rollout pseudo orders

        Returns:
            Action for the given power
        """
        raise NotImplementedError("Child classes must implement `format_output_seq_action`")

    def produce_action(self, game: pydipcc.Game, power: Power) -> Action:
        """
        Queries an agent to select most probable orders from the possible ones.

        Args:
            game: dipCC game object
            power: name of the power for which orders are queried.

        Returns:
            orders: list of orders for each position or list of build/disband orders for chosen positions.
        """
        seq = self.format_input_seq(game, power)
        raw_pred = self.get_model_pred(seq)["text"]
        orders = tuple(self.format_output_seq_action(raw_pred, power, game.current_short_phase))

        return orders

    def produce_many_action(
        self, game: pydipcc.Game, power: Power, num_preds: int, batch_size: Optional[int] = None,
    ) -> List[Tuple[Action, float]]:
        """
        Produce multiple orders by beam search.
        """
        seq = self.format_input_seq(game, power)
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size)
        orders = [
            (
                tuple(self.format_output_seq_action(raw_pred, power, game.current_short_phase)),
                score,
            )
            for raw_pred, score in many_raw_pred
        ]

        return orders

    def produce_joint_action(self, game: pydipcc.Game, power: Power) -> JointAction:
        """
        Return all orders from the perspective a single power

        This ParlAI agent was trained to predict all orders from the perspective of a single power.
        Instead of returning a single predicted order for that power, we return all predicted orders.
        This is useful for pseudo order agents.
        """
        raise NotImplementedError("`produce_joint_action` must be implemented by child class")

    def produce_many_joint_action(
        self, game: pydipcc.Game, power: Power, num_preds: int, batch_size: Optional[int] = None,
    ) -> JointPolicy:
        """
        Return multiple joint actions by beam search. Each result is joint action from the perspective of a single power.
        """
        raise NotImplementedError("`produce_many_joint_action` must be implemented by child class")

    def produce_rollout_joint_action_bilateral(
        self, game: pydipcc.Game, power: Power, recipient: Power
    ) -> RolloutJointAction:
        """
        Return a bilateral rollout joint action from the perspective a single power
        """
        raise NotImplementedError(
            "`produce_rollout_joint_action_bilateral` must be implemented by child class"
        )

    def score_candidate_actions(
        self,
        game: pydipcc.Game,
        candidates: List[Action],
        power: Power,
        target_power: Optional[Power] = None,
    ) -> List[Tuple[Action, float]]:
        candidate_seqs = self.format_candidate_seqs(candidates)
        assert len(candidate_seqs) == len(candidates)
        candidate_seqs_and_logps = dict(
            self.score_candidate_seqs(game, candidate_seqs, power, target_power=target_power)
        )

        # format_candidate_seqs and format_output_seq_action are NOT inverses!!
        # So carefully iterate over the original action-seq pairing and try to find
        # the logp of that sequence instead of doing the below:
        # return [
        #     (self.format_output_seq_action(candidate_seq, power, game.current_short_phase), logp,)
        #     for candidate_seq, logp in candidate_seqs_and_logps.items()
        # ]
        return _find_scoring_for_actions(candidates, candidate_seqs, candidate_seqs_and_logps)


class ParlAISingleOrderWrapper(BaseOrderWrapper):
    """
    Wrapper for models that predict a single action from the view of a single power.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert get_output_type(self.task_name) == "order"

    def format_output_seq_action(self, output_seq: str, power: Power, phase: Phase) -> Action:
        """
        Unflattens the output_seq corresponding to the given action.

        Args:
            output_seq: (str) output string produced by the model
            power: power taking the action; not necessary for the single order wrapper
            phase: current phase; not necessary for the single order wrapper

        Returns:
            Action for the given power
        """
        preds = self.formatter.orders_unflattener.unflatten_action(output_seq)
        return preds

    def format_candidate_seqs(self, candidates: List[Action], *args, **kwargs):
        """
        Given a list of candidates, format the output sequences
        """
        return [self.formatter.orders_flattener.flatten_action(x) for x in candidates]


class ParlAIAllOrderWrapper(BaseOrderWrapper):
    """
    Wrapper for models that predict a joint action from the view of a single power.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert get_output_type(self.task_name) == "allorder"

    def format_candidate_seqs(self, candidates: List[JointAction], power: Power) -> List[str]:
        """
        Given a list of candidates, format the output sequences
        """
        return [self.formatter.orders_flattener.flatten_joint_action(x, power) for x in candidates]

    def format_output_seq_action(self, output_seq: str, power: Power, phase: Phase) -> Action:
        """
        Given the output sequence corresponding to a joint action, return an action for a particular power.

        Args:
            output_seq: (str) output string produced by the model
            power: power taking the action
            phase: current phase

        Returns:
            Action for the given power
        """
        orders_dct = self.format_output_seq_joint_action(output_seq, phase)
        return orders_dct.get(power, ())

    def format_output_seq_joint_action(self, output_seq: str, phase: Phase) -> JointAction:
        """
        Given the output sequence corresponding to a joint action, return the joint action object.

        Args:
            output_seq: (str) output string produced by the model
            phase: current phase

        Returns:
            JointAction from the perspective of a power
        """
        orders_dct = self.formatter.orders_unflattener.unflatten_joint_action(output_seq)
        return orders_dct

    def produce_joint_action(self, game, view_of_power: Power) -> JointAction:
        """
        Predict a joint action from the perspective of a single power

        Args:
            game: game objection
            view_of_power: perspective of power
            recipient: (Optional) pseudo orders may be targeted towards a specific recipient

        Returns:
            JointAction from the view of a given power
        """
        seq = self.format_input_seq(game, view_of_power)
        raw_pred = self.get_model_pred(seq)["text"]
        all_orders = self.format_output_seq_joint_action(raw_pred, game.current_short_phase)

        return all_orders

    def produce_many_joint_action(
        self,
        game,
        view_of_power: Power,
        num_preds: int,
        batch_size: Optional[int] = None,
        recipient: Optional[Power] = None,
    ) -> JointPolicy:
        """
        Produce a joint policy from the perspective of a single power.
        """
        seq = self.format_input_seq(game, view_of_power, recipient)
        logging.debug(
            f"\n (ParlAIAllOrderWrapper.produce_many_joint_action) Input sequence:\n{seq}\n"
        )
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size)
        orders = [
            (self.format_output_seq_joint_action(raw_pred, game.current_short_phase), score)
            for raw_pred, score in many_raw_pred
        ]
        return orders


class MalformedPlausiblePseudoOrdersError(RuntimeError):
    pass


class ParlAIPlausiblePseudoOrdersWrapper(BaseOrderWrapper):
    """
    Wrapper for models used to predict plausible pseudo orders for a particular message.

    Can accommodate models trained to predict joint actions or rollout joint actions.
    """

    def __init__(self, *args, **kwargs):
        BaseOrderWrapper.__init__(self, *args, **kwargs)
        assert get_output_type(self.task_name) == "plausiblepseudoorder"
        assert self.opt["rollout_pseudo_orders"], "Only implemented for rollout pseudo orders"
        # Check how often we produce a rollout order without a movement phase
        self.total_nonmovement = 0
        self.movement_phase_missing_cnt = 0

    def expects_recipient(self) -> bool:
        """
        The plausible pseudo orders model expects to know the recipient a priori
        """
        return True

    def is_speaker_first(self) -> bool:
        return self.opt.get("speaker_first", False)

    def format_candidate_seqs(
        self, candidates: List[RolloutJointAction], power: Power, target_power: Power
    ):
        """
        Given a list of candidates, format the output sequences
        """
        if self.opt.get("rollout_phasemajor"):
            return [
                self.formatter.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                    x, power, target_power, speaker_first=self.is_speaker_first()
                )
                for x in candidates
            ]
        else:
            return [
                self.formatter.orders_flattener.flatten_rollout_joint_action_bilateral_powermajor(
                    x, power, target_power
                )
                for x in candidates
            ]

    def format_output_seq_action(self, output_seq: str, power: Power, phase: Phase) -> Action:
        """
        Given the output sequence corresponding to a joint action OR rollout action,
        return an action for a particular power.

        Args:
            output_seq: (str) output string produced by the model
            power: power taking the action
            phase: current phase

        Returns:
            Action for the given power
        """
        raise NotImplementedError(
            "Format output seq for a single power not yet implemented for the Plausible Orders Wrapper"
        )

    def format_output_seq_joint_action_bilateral(
        self, raw_pred: str, phase: Phase, game: pydipcc.Game
    ) -> JointAction:
        """
        Output is a bilateral joint action

        Args:
            output_seq: (str) output string produced by the model
            power: corresponds to the view of the particular power
            phase: current phase
            Game: game, just for error checking

        Returns:
            JointAction for the given power
        """

        rollout_action = self.format_output_seq_rollout_joint_action_bilateral(raw_pred, phase)
        if phase not in rollout_action:
            with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
                assert (
                    phase != game.current_short_phase
                ), f"{phase} not in {rollout_action} from {raw_pred}."

            # Corner case: Sometimes the PPO model thinks the game will end before a future
            # phase, so doesn't make a prediction for that future phase. To stay in distribution,
            # I think we really don't even want to include this phase in the pseudo-orders, but
            # since the API requires a JointAction for the phase, we will return empty orders :(
            #
            # In W1911A, France has 2 builds, but PPO model predicts empty orders for France with probability 1.
            #
            # If we prefix with 2 build for France, then we get
            #     raw_pred:
            #     FRANCE: A PAR B; F BRE B
            #     GERMANY:
            #     S1912M
            #     FRANCE:
            #     GERMANY:
            #     rollout_action: {'W1911A': {'FRANCE': ('F BRE B', 'A PAR B'), 'GERMANY': ()}, 'S1912M': {'FRANCE': (), 'GERMANY': ()}} .
            #
            # But if we prefix with empty builds for France, then we get
            #     raw_pred: W1911A
            #     FRANCE:
            #     GERMANY:
            #     rollout_action: {'W1911A': {'FRANCE': (), 'GERMANY': ()}}

            logging.warning(
                f"At {game.current_short_phase}, future phase {phase} not in {rollout_action} from {raw_pred}. Returning empty actions as fallback."
            )
            return {p: tuple() for p in POWERS}
        return rollout_action[phase]

    def format_output_seq_rollout_joint_action_bilateral(
        self, raw_pred: str, phase: Phase,
    ) -> RolloutJointAction:
        """
        Model was trained to predict a bilateral rollout joint action.

        Returns a rollout joint action
        """
        # BILATERAL rollout pseudo orders
        if self.opt.get("rollout_phasemajor"):
            return self.formatter.orders_unflattener.unflatten_rollout_joint_action_bilateral_phasemajor(
                raw_pred
            )
        else:
            return self.formatter.orders_unflattener.unflatten_rollout_joint_action_bilateral_powermajor(
                raw_pred, current_phase=phase,
            )

    def produce_joint_action_bilateral(
        self, game: pydipcc.Game, power: Power, recipient: Power
    ) -> JointAction:
        """
        Produce a rollout joint action for the power and recipient.

        Returns a rollout joint action
        """
        cur_phase = game.current_short_phase
        assert recipient is not None
        if self.opt.get("rollout_phasemajor"):
            game, rollout_joint_action = maybe_rollback_to_last_dialogue_phase(game)
            prefix_str = self.formatter.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                rollout_joint_action, power, recipient, speaker_first=self.is_speaker_first(),
            )
        else:
            assert_is_dialogue_phase(game)
            prefix_str = None

        seq = self.format_input_seq(game, power, recipient)
        raw_pred = self.get_model_pred(seq, prefix_str=prefix_str)["text"]
        joint_action = self.format_output_seq_joint_action_bilateral(raw_pred, cur_phase, game)
        return joint_action

    def _check_movement_phase_present(
        self,
        bilateral_rollout_joint_action: RolloutJointAction,
        curr_phase: Phase,
        power: Power,
        recipient: Power,
    ) -> RolloutJointAction:
        """
        Check that the rollout joint action contains orders up through the next movement phase.

        If next movement phase orders are not present, we add an empty action for each power
        """
        if is_movement_phase(curr_phase):
            return bilateral_rollout_joint_action

        def maybe_raise_err():
            if (
                self.movement_phase_missing_cnt > 5
                and self.movement_phase_missing_cnt / self.total_nonmovement > 0.1
            ):
                raise MalformedPlausiblePseudoOrdersError(
                    f"Movement phase corrupted in {self.movement_phase_missing_cnt}/{self.total_nonmovement} plausible rollout pseudo orders"
                )

        self.total_nonmovement += 1
        curr_year = int(curr_phase[1:-1])
        next_movement_phase = f"F{curr_year}M" if curr_phase[0] == "S" else f"S{curr_year + 1}M"
        if next_movement_phase not in bilateral_rollout_joint_action:
            # Movement phase is missing
            self.movement_phase_missing_cnt += 1
            # Raise runtime error if more than 5% of examples are missing the corresponding
            # movement phase
            maybe_raise_err()
            logging.warning(
                f"Movement phase not present in rollout joint action bilateral; adding empty orders for: {next_movement_phase}"
            )
            bilateral_rollout_joint_action[next_movement_phase] = {
                pow: tuple() for pow in [power, recipient]
            }
        else:
            if power not in bilateral_rollout_joint_action[next_movement_phase]:
                self.movement_phase_missing_cnt += 1
                maybe_raise_err()
                logging.warning(
                    f"Speaker missing from rollout joint action bilateral; adding empty orders for {power} in {next_movement_phase}"
                )
                bilateral_rollout_joint_action[next_movement_phase][power] = tuple()
            if recipient not in bilateral_rollout_joint_action[next_movement_phase]:
                self.movement_phase_missing_cnt += 1
                maybe_raise_err()
                logging.warning(
                    f"Recipient missing from rollout joint action bilateral; adding empty orders for {recipient} in {next_movement_phase}"
                )
                bilateral_rollout_joint_action[next_movement_phase][recipient] = tuple()

        return bilateral_rollout_joint_action

    def produce_rollout_joint_action_bilateral(
        self, game: pydipcc.Game, power: Power, recipient: Power
    ) -> RolloutJointAction:
        """
        Produce a rollout joint action for the power and recipient.

        Returns a rollout joint action
        """
        # we can produce a rollout action from a non dialogue phase: we teacher force the
        # phases up to the current phase, and predict the rest
        # (code copypasta'd from produce_joint_action_bilateral)

        cur_phase = game.current_short_phase

        assert recipient is not None
        if self.opt.get("rollout_phasemajor"):
            game, rollout_joint_action_prev = maybe_rollback_to_last_dialogue_phase(game)
            prefix_str = self.formatter.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                rollout_joint_action_prev, power, recipient, speaker_first=self.is_speaker_first(),
            )
            if prefix_str:
                prefix_str += "\n"
        else:
            assert_is_dialogue_phase(game)
            prefix_str = rollout_joint_action_prev = None

        assert recipient is not None
        seq = self.format_input_seq(game, power, recipient)
        raw_pred = self.get_model_pred(seq, prefix_str=prefix_str)["text"]
        rollout_joint_action = self.format_output_seq_rollout_joint_action_bilateral(
            raw_pred, cur_phase,
        )

        # Sometimes the model does not produce predictions through the next movement phase
        # If this happens, insert empty orders for the next movement phase
        rollout_joint_action = self._check_movement_phase_present(
            rollout_joint_action, cur_phase, power, recipient
        )

        # now, snip off any phases that we teacher-forced,
        # sanity checking that they match first

        if rollout_joint_action_prev is not None:
            for phase, joint_action_prev in rollout_joint_action_prev.items():
                for pwr, a in rollout_joint_action[phase].items():
                    with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
                        assert coast_unqualify_supports(
                            joint_action_prev[pwr]
                        ) == coast_unqualify_supports(
                            a
                        ), f"{rollout_joint_action_prev} != {rollout_joint_action} at {phase}, {pwr}"
                del rollout_joint_action[phase]

        return rollout_joint_action

    def produce_many_joint_action_bilateral(
        self,
        game,
        power: Power,
        num_preds: int,
        batch_size: Optional[int] = None,
        recipient: Optional[Power] = None,
    ) -> JointPolicy:
        """
        Produce a joint policy from the perspective of a single power.
        """
        cur_phase = game.current_short_phase
        assert recipient is not None
        if self.opt.get("rollout_phasemajor"):
            game, rollout_joint_action = maybe_rollback_to_last_dialogue_phase(game)
            prefix_str = self.formatter.orders_flattener.flatten_rollout_joint_action_bilateral_phasemajor(
                rollout_joint_action, power, recipient, speaker_first=self.is_speaker_first(),
            )
        else:
            assert_is_dialogue_phase(game)
            prefix_str = None

        seq = self.format_input_seq(game, power, recipient)
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size, prefix_str=prefix_str)
        orders = [
            (self.format_output_seq_joint_action_bilateral(raw_pred, cur_phase, game), score)
            for raw_pred, score in many_raw_pred
        ]
        return orders

    def score_candidate_rollout_order_tokens_bilateral(
        self,
        game: pydipcc.Game,
        candidates: List[JointAction],
        power: Power,
        target_power: Power,
        batch_size=10,
    ) -> List[Tuple[JointAction, Dict[Power, List[float]]]]:
        score_list = []
        for i in range(0, len(candidates), batch_size):  # chunk into 10 at a time to avoid OOM
            chunk = candidates[i : i + batch_size]
            score_list += self._score_candidate_rollout_order_tokens_bilateral_chunk(
                game, chunk, power, target_power
            )
        return score_list

    def _score_candidate_rollout_order_tokens_bilateral_chunk(
        self, game: pydipcc.Game, candidates: List[JointAction], power: Power, target_power: Power,
    ) -> List[Tuple[JointAction, Dict[Power, List[float]]]]:
        """
        Score candidate rollout orders, getting the per-order (conditional) logprobs.

        Returns: List of tuples, each containing
            - joint_action: The candidate action
            - order_logprobs: a dict from power to the logprob of each order, matching the logprobs in the joint_action.

        Note: We don't handle empty order strings here; there are no orders to score so sum(scores) == 0 by definition.
        """
        cur_phase = game.current_short_phase
        game, prev_phase_orders = maybe_rollback_to_last_dialogue_phase(game)
        rollout_candidates = [{**prev_phase_orders, cur_phase: c} for c in candidates]
        candidate_seqs = self.format_candidate_seqs(rollout_candidates, power, target_power)
        is_prefix = not cur_phase.endswith("M")
        if self.expected_rollout_type() == RolloutType.EXTENDED:
            is_prefix = is_prefix or game.current_short_phase == cur_phase

        if is_prefix:
            candidate_seqs = [s + "\n" for s in candidate_seqs]

        seq2cand = {s: c for s, c in zip(candidate_seqs, candidates)}

        # hardcode BPE-encodings of space and newline, assuming the gpt2 tokenizer.
        assert self.parlai_agent.opt["dict_tokenizer"] == "gpt2"
        BPE_SPACE = "\\xc4\\xa0"
        BPE_NEWLINE = "\\xc4\\x8a"
        end_token = self.parlai_agent.dict[self.parlai_agent.END_IDX]  # END_TOKEN = "__end__"
        order_delimiters = (";", BPE_NEWLINE, "[EO_O]", end_token)
        seq = self.format_input_seq(game, power, target_power=target_power)
        if not candidate_seqs:
            return []

        act = self.get_model_pred(seq, candidates=candidate_seqs)
        ret = []
        # n.b. text_candidates is not in the same order as candidate_seqs!
        for cand_seq, cand_scores in zip(act["text_candidates"], act["cand_scores"]):
            cand = seq2cand[cand_seq]

            # lol they removed the dash! gotta fix up here
            cand_vX = cand
            if self.formatter.orders_flattener.version >= 2:
                cand_vX = {pwr: tuple(x.replace(" - ", " ") for x in a) for pwr, a in cand.items()}

            # make a version with coast-unqualified supports because sometimes those come in
            cand_vX_coast_unqualified = {
                pwr: coast_unqualify_supports(a) for pwr, a in cand_vX.items()
            }

            # need a little FSM here
            cur_pwr = None
            cand_logprobs = {}
            cur_order, cur_logprob = "", 0.0

            def make_err_msg():
                return f"\npower: {power}\ntarget_power: {target_power}\nGame:\n{game.to_json()}\n\nseq:\n{seq}\n\ncandidates:\n{candidates}\ncand_seq\n{cand_seq}\n\ncand_scores:\n{cand_scores}\n\ncand_logprobs:\n{cand_logprobs}\n\nsample:\n{self.produce_rollout_joint_action_bilateral(game, power=power, recipient=target_power)}"

            for tok, score in cand_scores:
                if tok == ":":
                    assert score < 1.0, make_err_msg()
                elif tok.upper() in POWERS:
                    assert score < 1.0, make_err_msg()
                    assert cur_pwr is None
                    cur_pwr = tok.upper()
                    cur_order, cur_logprob = "", 0.0
                    cand_logprobs[cur_pwr] = [None for o in cand[cur_pwr]]
                elif cur_pwr is not None:  # decoding orders string
                    # slight hack: can't include end_token because we may not be predicting for all phases!
                    if tok != end_token or not is_prefix:
                        cur_logprob -= score
                    if tok in order_delimiters:
                        # technically we can back out the lexicographization, but I'm scared about bugs in that
                        # so I will actually reconstruct the order string and look it up
                        cur_order = cur_order.replace(BPE_SPACE, " ").strip()
                        if cur_order in cand_vX[cur_pwr]:
                            idx = cand_vX[cur_pwr].index(cur_order)
                            cand_logprobs[cur_pwr][idx] = cur_logprob
                        elif cur_order in cand_vX_coast_unqualified[cur_pwr]:
                            logging.warning(
                                f"Found {cur_order} in coast_unqualified supports only"
                            )
                            idx = cand_vX_coast_unqualified[cur_pwr].index(cur_order)
                            cand_logprobs[cur_pwr][idx] = cur_logprob
                        else:
                            with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
                                assert cur_order == "" or prev_phase_orders, (
                                    cur_order,
                                    cand_vX[cur_pwr],
                                )
                        cur_order, cur_logprob = "", 0.0
                    else:
                        cur_order += tok
                if tok == BPE_NEWLINE:
                    # assert cur_pwr is not None
                    cur_pwr = None

            # check that everything looks kosher
            # assert len(cand_logprobs) == 2
            assert set(cand_logprobs) == set([power, target_power]), make_err_msg()  # same powers

            all_logprobs_exist = all(
                p is not None for pwr_p in cand_logprobs.values() for p in pwr_p
            )
            if not all_logprobs_exist:
                logging.warning("==== ERROR: Missing logprob from orders ===")
                logging.warning(make_err_msg())
                cand_logprobs = {
                    pwr: [0 if p is None else p for p in pwr_p]
                    for pwr, pwr_p in cand_logprobs.items()
                }

            # add the logprobs for this candidate to the list
            ret.append((cand, cand_logprobs))

        return ret

    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Given a game object, format an input sequence to predict plausible pseudo orders
        from the view of a power and for a particular recipient

        Args:
            game: dipcc game object
            view_of_power: (str) power to get input for
            target_power: Power recipient to target pseudo orders towards
            timestamp: Unused in the orders wrappers; current timestamp

        Returns:
            str
        """
        assert target_power is not None

        format_output = self.formatter.change_format(
            game, self.input_format_str, self.metadata, view_of_power, recipient=target_power,
        )
        _, seqs = misc.last_dict_item(format_output)
        seq = seqs["input"]

        return seq


class ParlAIAllOrderIndependentWrapper(BaseOrderWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert get_output_type(self.task_name) == "allorderindependent"

    def _get_input_dict_for_speaker(self, game, view_of_power: Power) -> Dict[Power, str]:
        """
        Helper method which returns a set of inputs for a given speaker: one for each power
        """
        format_sequences = self.formatter.change_format(game, self.input_format_str, self.metadata)
        seqs = misc.last_dict_value(format_sequences)[view_of_power]
        power_to_seq = {seq["input"].split(" ")[-1][:-1].upper(): seq["input"] for seq in seqs}
        return power_to_seq

    def _format_input_seq(
        self,
        game,
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
        target_power: Power recipient to target orders towards
        timestamp: Unused in the orders wrappers, current timestamp

        Returns:
        A dict of `power` -> `input sequence for model`
        """
        power_to_seq = self._get_input_dict_for_speaker(game, view_of_power)
        assert target_power is not None
        assert target_power in power_to_seq

        return power_to_seq[target_power]

    def format_output_seq(self, output_seq: str) -> Action:
        preds = self.formatter.orders_unflattener.unflatten_action(output_seq)
        return preds

    def format_candidate_seqs(self, candidates: List[Action]):
        """
        Given a list of candidates, format the output sequences
        """
        return [self.formatter.orders_flattener.flatten_action(x) for x in candidates]

    def format_output_seq_joint_action(self, output_seq_dct) -> JointAction:
        """
        Returns all orders dict
        """
        formatted_all_order_dct = {}
        for k, v in output_seq_dct.items():
            formatted_all_order_dct[k] = self.formatter.orders_unflattener.unflatten_action(v)

        return formatted_all_order_dct

    def produce_action_for_target_power(
        self, game, view_of_power: Power, target_power: Power,
    ) -> Action:
        """
        Produce a single power's order from the perspective of another power (the "speaker").
        """
        seq = self.format_input_seq(game, view_of_power, target_power)
        raw_pred = self.get_model_pred(seq)["text"]
        return self.format_output_seq(raw_pred)

    def produce_action(self, game, power: Power) -> Action:
        """
        Override to support all order independent prediction
        """
        return self.produce_action_for_target_power(game, view_of_power=power, target_power=power)

    def produce_many_order_for_target_power(
        self,
        game,
        view_of_power: Power,
        target_power: Power,
        num_preds: int,
        batch_size: Optional[int] = None,
        *,
        two_powers_dialogue: Optional[Tuple[Power, Power]] = None,
    ) -> List[Tuple[Action, float]]:
        """
        Produce multiple orders by beam search.
        """
        metadata_copy = self.metadata.copy()
        if two_powers_dialogue:
            metadata_copy["two_powers_dialogue"] = two_powers_dialogue
        seq = self.format_input_seq(game, view_of_power, target_power)
        self.metadata = metadata_copy
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size)
        orders = [
            (tuple(self.format_output_seq(raw_pred)), score) for raw_pred, score in many_raw_pred
        ]

        return orders

    def produce_many_action(
        self, game, power: Power, num_preds: int, batch_size: Optional[int] = None,
    ) -> List[Tuple[Action, float]]:
        """
        Produce multiple orders by beam search.
        """

        return self.produce_many_order_for_target_power(
            game,
            view_of_power=power,
            target_power=power,
            num_preds=num_preds,
            batch_size=batch_size,
        )

    def produce_joint_action(self, game, power: Power) -> JointAction:
        power_to_seq = self._get_input_dict_for_speaker(game, power)
        all_order_dct = {
            pow: self.get_model_pred(inpt)["text"] for pow, inpt in power_to_seq.items()
        }
        all_orders = self.format_output_seq_joint_action(all_order_dct)

        return all_orders

    def score_candidate_actions(
        self,
        game: pydipcc.Game,
        candidates: List[Action],
        view_of_power: Power,
        target_power: Power,
    ) -> List[Tuple[Action, float]]:
        candidate_seqs = self.format_candidate_seqs(candidates)
        assert len(candidate_seqs) == len(candidates)
        candidate_seqs_and_logps = dict(
            self.score_candidate_seqs(game, candidate_seqs, view_of_power, target_power)
        )

        # format_candidate_seqs and format_output_seq are NOT inverses!!
        # So carefully iterate over the original action-seq pairing and try to find
        # the logp of that sequence instead of doing the below:
        # return [
        #     (self.format_output_seq(candidate_seq), logp)
        #     for candidate_seq, logp in candidate_seqs_and_logps
        # ]
        return _find_scoring_for_actions(candidates, candidate_seqs, candidate_seqs_and_logps)


def maybe_rollback_to_last_dialogue_phase(
    game: pydipcc.Game,
) -> Tuple[pydipcc.Game, RolloutJointAction]:
    """
    Handle the case that we are in a rollout phase, e.g. we are searching ahead
    to an M-phase from a previous R- or A-phase. Here last_dialogue_phase might
    be F1901R and game.current_short_phase might be S1902M. Since there are no
    messages for the phases since F1901R (asserted below), to keep the model
    in-disribution we must roll back to F1901R, predict a rollout action
    through S1902M, and extract the S1902M action to return. While decoding, we
    use prefix_str to force the model to decode the known orders for F1901R and
    W1901A.
    """
    last_dialogue_phase = game.get_metadata("last_dialogue_phase")

    if last_dialogue_phase and last_dialogue_phase != game.current_short_phase:
        # e.g. -> [<F1901R>, <W1901A>]
        phases_since = [
            p
            for p in game.get_phase_history()
            if sort_phase_key(p.name) >= sort_phase_key(last_dialogue_phase)
        ]
        # we could be more careful here by differentiating regular and extended rollouts
        assert sum(p.name.endswith("M") for p in phases_since) <= 1
        # current phase (S1902M) should have no messages
        assert len(game.messages) == 0, f"{last_dialogue_phase} != {game.current_short_phase}"
        # rollout phases since F1901R (W1901A) should have no messages
        assert all([len(p.messages) == 0 for p in phases_since[1:]])
        # get recipient actions for F1901R, W1901A
        rollout_joint_action = {p.name: p.orders for p in phases_since}
        # roll back game to F1901R for decoding
        return (
            game.rolled_back_to_phase_end(last_dialogue_phase),
            rollout_joint_action,
        )
    else:
        return game, {}


class ParlAIAllOrderIndependentRolloutWrapper(BaseOrderWrapper):
    def produce_many_order_for_target_power(
        self,
        game,
        view_of_power: Power,
        target_power: Power,
        num_preds: int,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[Action, float]]:
        cur_phase = game.current_short_phase
        game, rollout_joint_action = maybe_rollback_to_last_dialogue_phase(game)
        if rollout_joint_action:
            rollout_action = extract_rollout_action_for_power(rollout_joint_action, target_power)
            prefix_str = self.formatter.orders_flattener.flatten_rollout_action(rollout_action)
        else:
            prefix_str = None

        seq = self.format_input_seq(game, view_of_power, target_power)
        logging.debug(
            f"\n (ParlAIAllOrderIndependentRolloutWrapper.produce_many_order_for_target_power) Input sequence:\n{seq}\n"
        )
        many_raw_pred = self.get_model_pred_many(seq, num_preds, batch_size, prefix_str=prefix_str)

        orders = [
            (
                self.format_output_seq(raw_pred, current_phase=game.current_short_phase).get(
                    cur_phase, ()
                ),
                score,
            )
            for raw_pred, score in many_raw_pred
        ]
        return orders

    def format_output_seq(self, *args, **kwargs) -> RolloutAction:
        return self.formatter.orders_unflattener.unflatten_rollout_action(
            args[0], current_phase=kwargs.get("current_phase")
        )

    def score_candidate_actions(
        self,
        game: pydipcc.Game,
        candidates: List[Action],
        view_of_power: Power,
        target_power: Power,
    ) -> List[Tuple[Action, float]]:
        cur_phase = game.current_short_phase
        game, rollout_joint_action = maybe_rollback_to_last_dialogue_phase(game)
        target_rollout_action = extract_rollout_action_for_power(
            rollout_joint_action, target_power
        )
        assert cur_phase not in target_rollout_action
        candidate_rollout_actions = [
            {cur_phase: candidate, **target_rollout_action} for candidate in candidates
        ]
        candidate_seqs = [
            self.formatter.orders_flattener.flatten_rollout_action(x)
            for x in candidate_rollout_actions
        ]
        is_prefix = not cur_phase.endswith("M")

        if is_prefix:
            candidate_seqs = [s + "\n" for s in candidate_seqs]

        candidate_seqs_and_logps = dict(
            self.score_candidate_seqs(
                game,
                candidate_seqs,
                view_of_power,
                target_power,
                # must skip eos when we are scoring R- or A-phase orders, but the
                # model expects candidates to contain M-phase orders
                skip_end_token=is_prefix,
                skip_prefix=False,
            )
        )

        # format_candidate_seqs and format_output_seq are NOT inverses!!
        # So carefully iterate over the original action-seq pairing and try to find
        # the logp of that sequence instead of doing the below:
        # ret = [
        #     (
        #         self.format_output_seq(
        #             candidate_seq[:-1] if is_prefix else candidate_seq,  # remove newline
        #             current_phase=game.current_short_phase,
        #         ).get(cur_phase, ()),
        #         logp,
        #     )
        #     for candidate_seq, logp in candidate_seqs_and_logps
        # ]
        # return ret
        return _find_scoring_for_actions(candidates, candidate_seqs, candidate_seqs_and_logps)

    def _get_input_dict_for_speaker(self, game, view_of_power: Power) -> Dict[Power, str]:
        """
        Helper method which returns a set of inputs for a given speaker: one for each power
        """
        format_sequences = self.formatter.change_format(game, self.input_format_str, self.metadata)
        seqs = misc.last_dict_value(format_sequences)[view_of_power]
        power_to_seq = {seq["input"].split(" ")[-1][:-1].upper(): seq["input"] for seq in seqs}
        return power_to_seq

    def _format_input_seq(
        self,
        game,
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
        target_power: recipient to target orders towards
        timestamp: Unused in the orders wrappers -- current timestamp

        Returns:
        A dict of `power` -> `input sequence for model`
        """
        power_to_seq = self._get_input_dict_for_speaker(game, view_of_power)
        assert target_power is not None
        assert target_power in power_to_seq

        return power_to_seq[target_power]
