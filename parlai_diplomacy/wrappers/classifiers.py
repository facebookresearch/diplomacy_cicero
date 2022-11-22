#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
import logging
import string
import numpy as np
import random
from typing import Dict, Any, Tuple, Optional, List

from torch import LongTensor
import concurrent.futures

from parlai_diplomacy.tasks.draw_classifier.agents import DrawVoteStatus
from parlai_diplomacy.tasks.common_task_utils import CORRUPTED, REAL
from parlai_diplomacy.wrappers.base_wrapper import BaseWrapper
from parlai_diplomacy.wrappers.dialogue import (
    ParlAIResponseViewDialogueWrapper,
    BaseDialogueWrapper,
)
import parlai_diplomacy.utils.misc as misc
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import (
    add_message_to_game_copy,
    MessageObjectPart,
)
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME

from fairdiplomacy.utils.game import assert_game_from_view_of, get_game_draw_state
from fairdiplomacy.typedefs import MessageDict, Phase, Power, Timestamp
from fairdiplomacy import pydipcc
from fairdiplomacy.utils.sampling import normalize_p_dict
import fairdiplomacy.utils.parlai_multi_gpu_wrappers as parlai_multi_gpu_wrappers
from fairdiplomacy.pseudo_orders import PseudoOrders, RolloutType


class BaseClassifierWrapper(BaseWrapper):
    """
    Base wrapper for classifiers
    """

    OPT_OVERRIDE = {"interactive_mode": True, "print_scores": True, "load_from_checkpoint": False}

    def _format_input_seq(
        self,
        game: pydipcc.Game,
        view_of_power: Power,
        target_power: Optional[Power] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> str:
        """
        Most of our classifiers are based on dialogue models, so this is taken directly
        from the dialogue classifier wrapper.
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

    def _parse_classifier_output(self, output_seq: str) -> Tuple[str, float]:
        """
        Parse string classifier output to a prediction and associated probability
        """
        try:
            pred, prob = output_seq.split("\n")
            pred = pred.split("Predicted class: ")[-1]
            prob = float(prob.split("with probability: ")[-1])
        except Exception:
            raise RuntimeError(f"Incorrect output format of classifier: {output_seq}")

        return pred, prob

    def _get_class_probabilities(self, output_act: Dict) -> Tuple[List[str], LongTensor]:
        """
        Parse the returned output act to get class probabilities
        """
        probabilities = output_act["probs"].float()  # fp16 -> fp32
        classes = output_act["class_list"]

        return classes, probabilities

    def _sample_class(
        self, class_list: List[str], probabilities: List[float]
    ) -> Tuple[str, float, int]:
        """
        Sample class provided the probabilities
        """
        sample_index = int(np.random.choice(len(probabilities), 1, p=np.array(probabilities)))
        prob = float(probabilities[sample_index])
        pred = class_list[sample_index]

        return pred, prob, sample_index


class ParlAISleepClassifierWrapper(BaseClassifierWrapper):
    """
    Wrapper for ParlAI classifier agent that predicts how long to sleep

    Corresponds to task -t message_history_sleepclassifier_chunk
    """

    @property
    def task_name(self):
        """
        Override for backwards compatibility
        """
        task_name = self.opt["task"].split(":")[0]
        # Backwards compatibility for old models
        if task_name == "base_sleepclassifier_chunk":
            return "message_history_sleepclassifier_chunk"

        return task_name

    def expects_pseudo_orders(self) -> bool:
        # It doesn't seem necessary to use pseudo-orders for the sleep classifier.
        return False

    def is_sleepsix(self) -> bool:
        return "sleepsix" in self.opt["task"]

    def format_output_seq(
        self,
        output_act: Dict[Any, Any],
        power: Power,
        target_power: Optional[Power] = None,
        min_sleep: float = None,
        max_sleep: float = None,
        inf_threshold: float = 0,
    ) -> Tuple[Timestamp, float, str]:
        classes, probs = self._get_class_probabilities(output_act)

        # Band-aid to stop us ever sampling <pad>
        assert classes[0] == "pad"
        assert probs[0] < 0.01
        probs[0] = 0

        # Apply inf threshold
        assert classes[-1] == "inf", classes
        if probs[-1] < inf_threshold:
            logging.info(f"Prevent sampling inf, p={probs[-1]} < {inf_threshold}")
            probs[-1] = 0

        # sample the probabilities
        probs /= probs.sum()  # normalize probabilities prior to sampling
        pred_class, prob, sample_index = self._sample_class(classes, probs)
        # If the classes are something like [0, 10, 20, inf], for a prediction of '20', we take a random value in the range 10-20
        upper_bound = float(pred_class)
        lower_bound = (
            float(classes[sample_index - 1]) if sample_index > 1 else 0
        )  # Label 0 is "pad"
        pred = (
            random.randint(int(lower_bound), int(upper_bound))
            if upper_bound != float("inf")
            else upper_bound
        )

        classifier_pred = pred
        pred = max(pred, min_sleep) if min_sleep is not None else pred
        pred = min(pred, max_sleep) if max_sleep is not None else pred
        if pred != classifier_pred:
            logging.info(f"Overriding sleep classifier from {classifier_pred} to {pred}")

        # Add up to one second of noise
        rescaled = (
            Timestamp.from_seconds(pred + random.random())
            if pred != float("inf")
            else INF_SLEEP_TIME  # inf ~ 1000 years
        )

        return rescaled, prob, pred_class

    def get_sleep_time(
        self,
        game: pydipcc.Game,
        power: Power,
        target_power: Optional[Power] = None,
        *,
        inf_threshold: float = 0,
    ) -> Timestamp:
        if "sleepsix" in self.opt["task"]:
            assert target_power is not None
            return self.get_sleepsix_times(
                game, power, [target_power], inf_thresholds=[inf_threshold]
            )[target_power][0]
        else:
            return self.get_legacy_sleep_time(game, power, inf_threshold=inf_threshold)

    def get_legacy_sleep_time(
        self, game: pydipcc.Game, power: Power, *, inf_threshold: float = 0,
    ) -> Timestamp:
        """
        Sample from a sleep time distribution.

        inf will only be sampled if p(inf) >= inf_threshold
        """
        assert "sleepsix" not in self.opt["task"], self.opt["task"]
        seq = self.format_input_seq(
            game, power, timestamp=None,
        )  # Timestamp is set to None here, since we are predicting the timestamp
        delim = " " if self.opt.get("task_version", 1) < 2 else "\n"
        assert f"0{delim}{game.current_short_phase}" in seq
        act = self.get_model_pred(seq)
        # format_output_seq samples from the distribution in act
        sleep_time, _, _ = self.format_output_seq(act, power, inf_threshold=inf_threshold)
        return sleep_time

    def get_sleepsix_distributions(
        self, game: pydipcc.Game, power: Power, target_powers: List[Power],
    ) -> Dict[Power, List[float]]:
        return {
            target_power: self.get_model_pred(
                self.format_input_seq(game, power, target_power=target_power, timestamp=None)
            )["probs"]
            .float()
            .tolist()
            for target_power in target_powers
        }

    def get_sleepsix_times(
        self,
        game: pydipcc.Game,
        power: Power,
        target_powers: List[Power],
        *,
        inf_thresholds: Optional[List[float]] = None,
    ) -> Dict[Power, Tuple[Timestamp, float]]:
        """Return values are (sampled timestamp, sample probability)"""
        assert "sleepsix" in self.opt["task"], self.opt["task"]
        if inf_thresholds is None:
            inf_thresholds = [0 for _ in target_powers]
        sleep_times, sleep_time_ps, pred_classes = {}, {}, {}
        for target_power, inf_threshold in zip(target_powers, inf_thresholds):
            seq = self.format_input_seq(
                game, power, target_power=target_power, timestamp=None,
            )  # Timestamp is set to None here, since we are predicting the timestamp
            delim = " " if self.opt.get("task_version", 1) < 2 else "\n"
            assert f"0{delim}{game.current_short_phase}" in seq
            logging.debug(f"\n(SleepSixWrapper) Input sequence:\n{seq}\n")
            act = self.get_model_pred(seq)
            # format_output_seq samples from the distribution in act
            rescaled_time, prob, pred_class = self.format_output_seq(
                act, power, target_power=target_power, inf_threshold=inf_threshold
            )
            sleep_times[target_power] = (rescaled_time, prob)
            pred_classes[target_power] = pred_class
            classes, sleep_time_ps[target_power] = self._get_class_probabilities(act)

        self.log_sleep_times(power, classes, sleep_times, sleep_time_ps, pred_classes)  # type: ignore

        return sleep_times

    @staticmethod
    def log_sleep_times(
        power: Power,
        classes: List[str],
        sleep_times: Dict[Power, Timestamp],
        sleep_time_ps: Dict[Power, List[float]],
        pred_classes: Dict[Power, str],
    ):
        """Info-logs a table of sleep times for all targets

        - Under 100-char width
        - Emphasizes the class which was sampled for each target
        """
        # strip PAD class
        classes = classes[1:]
        sleep_time_ps = {p: ps[1:] for p, ps in sleep_time_ps.items()}

        # get idx for predicted class for each power
        pred_class_idxs = {p: classes.index(pred_class) for p, pred_class in pred_classes.items()}

        # headers are classes, shortened to fit in 5 chars
        headers = ["tgt"] + [
            c if len(c) < 5 else str(round(float(c) / 1000)) + "k" for c in classes
        ]

        def emphasize_if(x: str, should_star: bool) -> str:
            if not should_star:
                return x
            assert x[0] == " " and x[-1] == " ", x
            return "|" + x[1:-1] + "|"

        val_lists = [
            [target[:3] + " "]
            + [
                emphasize_if(
                    f"{p:.2f} ".replace("0.", " .").replace("1.00", " 1.0"),
                    i == pred_class_idxs[target],
                )
                for i, p in enumerate(probs)
            ]
            for target, probs in sleep_time_ps.items()
        ]
        rows = [
            "".join(h.center(5) for h in headers),
            " --- " * len(headers),
            *["".join(val_list) for val_list in val_lists],
        ]
        table = "\n".join(r.lstrip() for r in rows)

        logging.info(f"{power} sleep time distributions:\n{table}")


class ParlAIRecipientClassifierWrapper(BaseClassifierWrapper):
    """
    Wrapper for ParlAI classifier agent that predicts who to speak to next
    """

    def format_output_seq(self, distribution: Dict[Power, float], power: Power) -> Power:
        """
        Given the output act from the model, sample a recipient
        """
        pred, prob, _ = self._sample_class(list(distribution.keys()), list(distribution.values()))
        logging.info(f"{power} should speak to {pred.upper()} with probability {prob}")
        return pred.upper()

    def get_recipient(self, game: pydipcc.Game, power: Power, timestamp: Timestamp) -> Power:
        """
        Returns a recipient predicted for a message from power
        """
        recipient_distribution = self.get_recipient_distribution(game, power, timestamp)
        return self.format_output_seq(recipient_distribution, power)

    def get_recipient_distribution(
        self, game: pydipcc.Game, power: Power, timestamp: Timestamp
    ) -> Dict[Power, float]:
        """
        Returns a set of probabilities corresponding to possible recipients
        """
        seq = self.format_input_seq(game, power, timestamp=timestamp)
        output_act = self.get_model_pred(seq)
        classes, probabilities = self._get_class_probabilities(output_act)

        distribution = {c.upper(): float(p) for c, p in zip(classes, probabilities)}
        logging.info(f"{power} recipient classifier distribution: {distribution}")

        # Filter to only alive powers and re-normalize
        alive_powers = game.get_alive_powers()
        distribution = {
            recip: (float(p) if (recip in alive_powers and recip != power) else 0.0)
            for recip, p in distribution.items()
        }

        # Block ALL messages -- Recipient classifiers predicting draw votes is DEPRECATED: see #1380
        if "ALL" in distribution and distribution["ALL"] > 0.0:
            logging.warning(
                f"Squashing draw probability {distribution['ALL']} to 0.0, recipient classifier draw behavior is not supported"
            )
            distribution["ALL"] = 0.0

        return normalize_p_dict(distribution)


class ParlAIDrawClassifierWrapper(BaseClassifierWrapper):
    """
    Wrapper for ParlAI classifier agent that predicts whether or not to
        - Draw
        - Unvote for a draw
        - Neither
    """

    def get_draw_vote_status(
        self, game: pydipcc.Game, power: Power, timestamp: Timestamp
    ) -> DrawVoteStatus:
        """
        Get the possible draw vote status
        """
        distribution = self.get_draw_distribution(game, power, timestamp)

        # Zero out impossible states
        game_draw_state = get_game_draw_state(game)
        powers_have_drawn = [k for k, v in game_draw_state.items() if v]
        if powers_have_drawn:
            logging.info(f"Current draw votes: {', '.join(powers_have_drawn)}")
        power_prev_draw_status = game_draw_state[power]
        if power_prev_draw_status:
            # Power has already voted for a draw
            distribution[DrawVoteStatus.DRAW.value] = 0.0
        else:
            distribution[DrawVoteStatus.UNDRAW.value] = 0.0
        distribution = normalize_p_dict(distribution)

        # Sample a possible class
        pred, _, _ = self._sample_class(list(distribution.keys()), list(distribution.values()))

        return DrawVoteStatus[pred]

    def get_draw_distribution(
        self, game: pydipcc.Game, power: Power, timestamp: Timestamp
    ) -> Dict[str, float]:
        """
        Returns a set of probabilities corresponding to possible draw states
        """
        seq = self.format_input_seq(game, power, timestamp=timestamp)
        output_act = self.get_model_pred(seq)
        classes, probabilities = self._get_class_probabilities(output_act)

        distribution = {c: float(p) for c, p in zip(classes, probabilities)}
        logging.info(f"{power} draw status classifier distribution: {distribution}")

        return distribution


class BaseNonsenseClassifierWrapper(ABC):
    @abstractmethod
    def should_run_classifier(
        self, game: pydipcc.Game, msg: MessageDict, classifier_name: str
    ) -> bool:
        """
        Returns True/False corresponding to whether the classifier should be run for a given example

        True by default for all classifiers, subclasses can override with more specific criteria.
        """
        raise NotImplementedError

    @abstractmethod
    def maybe_lowercase_judgement_message(self, potential_msg: MessageDict) -> MessageDict:
        raise NotImplementedError

    @abstractmethod
    def get_nonsense_pred_dist(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Get the output distribution of the classifier given a game and a potential message that the sender intends to send
        Args:
            game: a Game object from the view of the nonsense sender
            potential_msg: the potential nonsensical message
        Returns:
            prediction: str, the string representing model's prediction
            prediction_prob: float, probability assigned to model's prediction
            class_probs: Dict[str, float], dict of each class and its associated probability.
        """
        raise NotImplementedError

    @abstractmethod
    def get_corrupted_prob_with_threshold(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_corrupted_prob_with_threshold_if_should_run(
        self, game: pydipcc.Game, potential_msg: MessageDict, classifier_name: str
    ) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_candidate_logprobs(
        self,
        act: Dict[str, Any],
        skip_prefix: Optional[bool] = False,
        skip_end_token: Optional[bool] = False,
    ) -> List[float]:
        """
        As an example return max token loss per candidate eturn max token loss
        """
        raise NotImplementedError

    @abstractmethod
    def get_nonsense_status(self, game: pydipcc.Game, potential_msg: MessageDict) -> bool:
        """
        Get the nonsense status
        game: a Game objection from the view of the nonsense_sender
        potential_msg: the potential nonsensical message
        returns:
        is_nonsense boolean
        """
        raise NotImplementedError


class ParlAINonsenseClassifierWrapper(BaseClassifierWrapper, BaseNonsenseClassifierWrapper):
    """
    Wrapper for ParlAI classifier agent that predicts if last message in the context belongs to one of the synthetic nonsense categories
    """

    def __init__(self, model_path: str, additional_args=None):
        super().__init__(model_path, additional_args)
        self.threshold = additional_args.get("overrides", {}).get("threshold", 0.5)  # type: ignore

    def should_run_classifier(
        self, game: pydipcc.Game, msg: MessageDict, classifier_name: str
    ) -> bool:
        """
        Returns True/False corresponding to whether the classifier should be run for a given example

        True by default for all classifiers, subclasses can override with more specific criteria.
        """
        return True

    @property
    def task_name(self):
        """
        Nonsense classifier trained on multiple tasks so we split on ',' and strip the first token of the task_name
        """
        task_name = "_".join(self.opt["task"].split(",")[0].split("_")[1:])
        return task_name

    def validate_input(self, input_seq: str):
        pass

    def format_output_seq(
        self, output_act: Dict, power: Power
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Format the output_act dictionary into objects needed for logging
        Args:
            output_act: Dict
            power: Power
        Returns:
            prediction: str, the string representing model's prediction
            prediction_prob: float, probability assigned to model's prediction
            class_probs: Dict[str, float], dict of each class and its associated probability.
        """
        classes, probabilities = self._get_class_probabilities(output_act)
        prediction_prob, prediction = max(zip(probabilities, classes))
        class_probs = {c: p.item() for c, p in zip(classes, probabilities)}
        return prediction, prediction_prob.item(), class_probs

    def maybe_lowercase_judgement_message(self, potential_msg: MessageDict) -> MessageDict:
        if self.opt.get("lowercase_judgement_message", False):
            potential_msg[MessageObjectPart.MESSAGE] = potential_msg[
                MessageObjectPart.MESSAGE
            ].lower()
        return potential_msg

    def get_nonsense_pred_dist(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Get the output distribution of the classifier given a game and a potential message that the sender intends to send
        Args:
            game: a Game object from the view of the nonsense sender
            potential_msg: the potential nonsensical message
        Returns:
            prediction: str, the string representing model's prediction
            prediction_prob: float, probability assigned to model's prediction
            class_probs: Dict[str, float], dict of each class and its associated probability.
        """

        sender = potential_msg[MessageObjectPart.SENDER]
        assert_game_from_view_of(game, sender)
        seq = self.format_input_seq(
            game,
            sender,
            target_power=potential_msg[MessageObjectPart.RECIPIENT],
            timestamp=potential_msg[MessageObjectPart.TIME_SENT],
        )  # produces a sequence of messages along with state (if applicable).

        potential_msg = self.maybe_lowercase_judgement_message(potential_msg)
        msg_seq = self.formatter.messagehistory_flattener.flatten_phase_messages(
            [potential_msg]
        )  # format the potential messages separately
        seq = "\n".join(
            (seq, msg_seq)
        )  # concate the separately formatted message with the rest of the message sequence
        """
        Expected format for seq
        PHASE
        A -> B message1 [EO_M]
        B -> A message2 [EO_M]
        ...
        A -> B messagen PHASE B 5:
        PHASE
        B -> A potential message [EO_M]
        """
        output_act = self.get_model_pred(seq)
        return self.format_output_seq(output_act, sender)

    def get_corrupted_prob_with_threshold(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[float, float]:
        (_, _, classes_probs,) = self.get_nonsense_pred_dist(game, potential_msg)
        return (classes_probs[CORRUPTED], self.threshold)

    def get_corrupted_prob_with_threshold_if_should_run(
        self, game: pydipcc.Game, potential_msg: MessageDict, classifier_name: str
    ) -> Tuple[float, float]:
        if not self.should_run_classifier(game, potential_msg, classifier_name):
            # Don't run classifier, so return dummy nonsense and threshold
            return (0.0, 1.0)
        return self.get_corrupted_prob_with_threshold(game, potential_msg)

    def get_nonsense_status(self, game: pydipcc.Game, potential_msg: MessageDict) -> bool:
        """
        Get the nonsense status
        game: a Game objection from the view of the nonsense_sender
        potential_msg: the potential nonsensical message
        returns:
        is_nonsense boolean
        """
        _, _, classes_probs = self.get_nonsense_pred_dist(game, potential_msg)
        if classes_probs[CORRUPTED] >= self.threshold:
            return True
        return False


class ParlAIHumanVsModelClassifierWrapper(ParlAINonsenseClassifierWrapper):
    """
    Wrapper for ParlAI classifier agent that predicts if last message in the context is human generated (acceptable) or model generated (not acceptable)
    """

    def __init__(self, model_path: str, additional_args=None):
        super().__init__(model_path, additional_args)
        self.threshold = additional_args.get("overrides", {}).get("threshold", 0.5)  # type: ignore

    def should_run_classifier(
        self, game: pydipcc.Game, msg: MessageDict, classifier_name: str
    ) -> bool:
        """
        Returns True/False corresponding to whether the model is an HVM classifier
        """

        if self.should_never_filter_short_messages(classifier_name):
            raw_text = msg["message"]
            for punc in list(string.punctuation):
                raw_text = raw_text.replace(punc, " ")
            raw_text_tokens = [tok for tok in raw_text.split(" ") if tok != ""]

            if len(raw_text_tokens) <= 5:
                logging.info(
                    f"Skipping nonsense classifier {classifier_name} for message proposal {msg['message']}, due to it being too short."
                )
                return False

        ### Filters for S1901M

        if game.current_short_phase != "S1901M":
            # HVM classifiers should always be run on phases > S1901M
            return True

        if self.requires_message_history():
            # Check that there non-sender messages for the model to rely on
            sender = msg[MessageObjectPart.SENDER]
            recipient = msg[MessageObjectPart.RECIPIENT]
            non_sender_messages = [
                x for x in game.messages.values() if x[MessageObjectPart.SENDER] != sender
            ]
            if not non_sender_messages:
                # No messages for the model to rely on
                logging.info(
                    f"Skipping nonsense classifier {classifier_name} for message proposal {msg['message']}, due to it not having message history to rely on."
                )
                return False

            if self.requires_bilateral_message_history():
                recipient_messages = [
                    x for x in game.messages.values() if x[MessageObjectPart.SENDER] == recipient
                ]
                if not recipient_messages:
                    logging.info(
                        f"Skipping nonsense classifier {classifier_name} for message proposal {msg['message']}, due to not having bilateral message history to rely on."
                    )
                    return False
        elif self.requires_order_history():
            logging.info(
                f"Skipping nonsense classifier {classifier_name} for message proposal {msg['message']}, due to it not having order history to rely on."
            )
            return False

        # Model doesn't rely on message history OR has some message history available to use
        return True

    @property
    def task_name(self):
        return self.opt["task"]

    def should_never_filter_short_messages(self, classifier_name):
        return "justification" in classifier_name

    def requires_message_history(self) -> bool:
        """
        Whether or not the input contains message history
        """
        return "message_history" in self.task_name

    def requires_bilateral_message_history(self) -> bool:
        """
        Whether or not the message history in the input is bilateral
        """
        if not self.requires_message_history():
            return False

        return self.opt.get("2person_dialogue", False) or self.opt.get("two_party_dialogue", False)

    def requires_order_history(self) -> bool:
        """
        Whether or not the input contains order history
        """
        return "orderhistory" in self.task_name

    def get_nonsense_pred_dist(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Get the output distribution of the classifier given a game and a potential message that the sender intends to send
        Args:
            game: a Game object from the view of the nonsense sender
            potential_msg: the potential nonsensical message
        Returns:
            prediction: str, the string representing model's prediction
            prediction_prob: float, probability assigned to model's prediction
            class_probs: Dict[str, float], dict of each class and its associated probability.
        """

        sender = potential_msg[MessageObjectPart.SENDER]
        assert_game_from_view_of(game, sender)
        seq = self.format_input_seq(
            game,
            sender,
            target_power=potential_msg[MessageObjectPart.RECIPIENT],
            timestamp=potential_msg[MessageObjectPart.TIME_SENT],
        )  # produces a sequence of messages along with state (if applicable).
        seq = self.formatter.generate_input(seq, potential_msg)  # type: ignore (type system is not good at parsing factory-generated objects)
        output_act = self.get_model_pred(seq)
        return self.format_output_seq(output_act, sender)


class EnsembleNonsenseClassifierWrapper:
    def __init__(
        self,
        nonsense_classifier_executors: Dict[str, parlai_multi_gpu_wrappers.ParlaiExecutor] = {},
    ):
        self.classifier_executors = nonsense_classifier_executors
        self.classifiers_expects_PO = {
            name: classifier_executor.compute("expects_pseudo_orders", None).result()
            for name, classifier_executor in self.classifier_executors.items()
        }
        self.classifiers_expected_rollout_type = {
            name: classifier_executor.compute("expected_rollout_type", None).result()
            for name, classifier_executor in self.classifier_executors.items()
        }
        self.classifier_thresholds = {
            name: classifier_executor.get("threshold").result()
            for name, classifier_executor in self.classifier_executors.items()
        }

        logging.info(
            f"Classifiers in ensemble: "
            + str(
                {
                    name: classifier_executor.get_model_opt()["task"]
                    for name, classifier_executor in self.classifier_executors.items()
                }.items()
            )
        )

    def expects_pseudo_orders(self) -> bool:
        return any(self.classifiers_expects_PO.values())

    def update_pseudo_orders(self, phase: Phase, speaker: Power, pseudo_orders: PseudoOrders):
        for classifier_name, classifier_executor in self.classifier_executors.items():
            if self.classifiers_expects_PO[classifier_name]:
                pseudo_orders_for_cur_classifier = pseudo_orders
                if not pseudo_orders.check_rollout(
                    self.classifiers_expected_rollout_type[classifier_name]
                ):
                    # For now, asssume this means we are trying to change EXTENDED rollout PO to RA_ONLY rollout PO
                    if (
                        pseudo_orders.check_rollout(RolloutType.EXTENDED)
                        and self.classifiers_expected_rollout_type[classifier_name]
                        == RolloutType.RA_ONLY
                    ):
                        pseudo_orders_for_cur_classifier = (
                            pseudo_orders.as_rollout_except_movement_action()
                        )

                    else:
                        raise RuntimeError(
                            "Not all pseudo-order formatting discrepancies can be resolved at inference time."
                        )

                classifier_executor.compute(
                    "update_pseudo_orders", None, phase, speaker, pseudo_orders_for_cur_classifier
                )

    def get_corrupted_prob_with_threshold_from_classifier(
        self, classifier_name: str, game: pydipcc.Game, potential_msg: MessageDict
    ) -> concurrent.futures.Future:
        if self.classifier_executors[classifier_name]:
            return self.classifier_executors[classifier_name].compute(
                "get_corrupted_prob_with_threshold_if_should_run",
                game,
                potential_msg,
                classifier_name,
            )
        else:
            return parlai_multi_gpu_wrappers.InstantFuture((0, 1))

    def get_verbose_nonsense_status(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:

        classifiers_outputs: Dict[str, Dict[str, float]] = {}

        async_model_calls = {}
        for name in self.classifier_executors:
            async_model_calls[name] = self.get_corrupted_prob_with_threshold_from_classifier(
                name, game, potential_msg
            )

        for name in self.classifier_executors:
            p_nonsense, threshold = async_model_calls[name].result()
            classifiers_outputs[name] = {"p_nonsense": p_nonsense, "threshold": threshold}

        is_nonsense = any(
            [
                output["p_nonsense"] >= output["threshold"]
                for output in classifiers_outputs.values()
            ]
        )

        res = {name: output for name, output in classifiers_outputs.items()}

        logging.info(
            f"EnsembleNonsenseClassifierWrapper: message: {potential_msg['message']} phase:{potential_msg['phase']} time_sent:{potential_msg['time_sent']}: "
            f"{res}"
        )

        return is_nonsense, res

    def get_nonsense_status(self, game: pydipcc.Game, potential_msg: MessageDict) -> bool:
        """
        Get the nonsense status
        game: a Game objection from the view of the nonsense_sender
        potential_msg: the potential nonsensical message
        returns:
        is_nonsense boolean
        """
        is_nonsense, status = self.get_verbose_nonsense_status(game, potential_msg)
        return is_nonsense


class SludgeDialogueAsNonsenseClassifierWrapper(BaseDialogueWrapper):
    """
    Using dialogue model as a nonsense classifier
    """

    def __init__(self, model_path: str, additional_args={}):
        super().__init__(model_path, additional_args)
        self.threshold = additional_args.get("overrides", {}).get("threshold", 0.5)  # type: ignore

    def should_run_classifier(
        self, game: pydipcc.Game, msg: MessageDict, classifier_name: str
    ) -> bool:
        """
        Returns True/False corresponding to whether the classifier should be run for a given example

        True by default for all classifiers, subclasses can override with more specific criteria.
        """
        return True

    def maybe_lowercase_judgement_message(self, potential_msg: MessageDict) -> MessageDict:
        return potential_msg

    def get_nonsense_pred_dist(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Get the output distribution of the classifier given a game and a potential message that the sender intends to send
        Args:
            game: a Game object from the view of the nonsense sender
            potential_msg: the potential nonsensical message
        Returns:
            prediction: str, the string representing model's prediction
            prediction_prob: float, probability assigned to model's prediction
            class_probs: Dict[str, float], dict of each class and its associated probability.
        """

        self.game = game
        phase = game.current_short_phase
        candidates = [potential_msg[MessageObjectPart.MESSAGE]]

        candidate_seqs = self.format_candidate_seqs(
            candidates,
            potential_msg[MessageObjectPart.SENDER],
            potential_msg[MessageObjectPart.RECIPIENT],
            phase,
            add_eom_token=False,
        )

        max_token_scores = self.score_candidate_seqs(
            game,
            candidate_seqs,
            potential_msg[MessageObjectPart.SENDER],
            potential_msg[MessageObjectPart.RECIPIENT],
            potential_msg[MessageObjectPart.TIME_SENT],
            False,
            True,
        )
        assert len(max_token_scores) == 1  # using it as a nonsense classifier, assumes bsz 1
        score = max_token_scores[0][1]

        is_nonsense = CORRUPTED if score > self.threshold else REAL
        class_probs = {CORRUPTED: score}

        return is_nonsense, score, class_probs  # type: ignore

    def get_corrupted_prob_with_threshold(
        self, game: pydipcc.Game, potential_msg: MessageDict
    ) -> Tuple[float, float]:
        (_, _, classes_probs,) = self.get_nonsense_pred_dist(game, potential_msg)
        return (classes_probs[CORRUPTED], self.threshold)

    def get_corrupted_prob_with_threshold_if_should_run(
        self, game: pydipcc.Game, potential_msg: MessageDict, classifier_name: str
    ) -> Tuple[float, float]:
        if not self.should_run_classifier(game, potential_msg, classifier_name):
            # Don't run classifier, so return dummy nonsense and threshold
            return (0.0, 1.0)
        return self.get_corrupted_prob_with_threshold(game, potential_msg)

    def get_candidate_logprobs(
        self,
        act: Dict[str, Any],
        skip_prefix: Optional[bool] = False,
        skip_end_token: Optional[bool] = False,
    ) -> List[float]:
        """
        As an example return max token loss per candidate eturn max token loss
        """
        max_token_scores = []
        for per_token_losses in act["cand_scores"]:
            max_token_score = max([x[1] for x in per_token_losses])
            max_token_scores.append(max_token_score)

        return max_token_scores

    def get_nonsense_status(self, game: pydipcc.Game, potential_msg: MessageDict) -> bool:
        """
        Get the nonsense status
        game: a Game objection from the view of the nonsense_sender
        potential_msg: the potential nonsensical message
        returns:
        is_nonsense boolean
        """
        _, _, classes_probs = self.get_nonsense_pred_dist(game, potential_msg)
        if classes_probs[CORRUPTED] >= self.threshold:
            return True
        return False
