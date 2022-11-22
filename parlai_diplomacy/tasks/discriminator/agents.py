#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from fairdiplomacy.typedefs import Phase, Power
import jsonlines
import math
import random
import os
from copy import deepcopy
from fairdiplomacy.models.consts import POWERS
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.loader import register_teacher
import parlai.utils.logging as logging

from parlai_diplomacy.tasks.dialogue.base_agent import BaseDialogueChunkTeacher
from parlai_diplomacy.utils.game2seq.format_helpers.misc import load_json
from parlai_diplomacy.tasks.discriminator import change_entity_utils

from parlai_diplomacy.tasks.common_task_utils import (
    CORRUPTED,
    REAL,
    INCORRECT_RECIPIENT,
    CORRUPTED_ENTITY,
    INCORRECT_PHASE,
    INCORRECT_GAME,
    REAPEATED_MESSAGE,
)
from parlai_diplomacy.utils import datapath_constants
from typing import List, Optional
from parlai_diplomacy.utils.game2seq.format_helpers.message_history import MessageObjectPart

"""
Teachers for generating real or corrupted (noisy) examples.
We use the labels for training a discriminative classifier
"""

# The number of randomly picked messges from the previously seen game examples
# to keep for swapping in the future games.
GAME_CACHE_BUFFER_SIZE = 1024
DEFAULT_MAX_VALID_EXAMPLE_PER_TYPE = 4_000
MAX_AVAILABLE_VALID_EXAMPLES_PER_TYPE = 12_758


def insert_discriminator_label(message: str, label: str) -> str:
    """
    Takes a message string which can have 1 or more [EO_M] tokens, and adds a
    label token after each [EO_M] token. This method does not change
    the newline characters in the string.
    """
    new_message = []
    for line in message.splitlines():
        new_line = []
        for i in line.split():
            if i == "[EO_M]":
                i = " ".join([i, label])
            new_line += [i]
        new_message += [" ".join(new_line)]
    return "\n".join(new_message)


def get_powers(exclude: List[str] = None):
    if not exclude:
        exclude = []
    return [p.capitalize() for p in POWERS if p not in [e.upper() for e in exclude]]


def _get_valid_dpath(teacher_corruption_type, dformat=None):
    if "unittests:" in teacher_corruption_type:  # Path to data used in unittests
        unittests_root_dir = teacher_corruption_type.split(":")[1]
        return os.path.join(unittests_root_dir, datapath_constants.DISCRIMINATOR_VALID_TEST_DATA)
    return os.path.join(
        datapath_constants.DISCRIMINATOR_VALID_DATA_ROOT,
        teacher_corruption_type,
        f"{dformat}.jsonl",
    )


def _get_seq_valid_dpath(teacher_corruption_type, dformat=None):
    if "unittests:" in teacher_corruption_type:  # Path to data used in unittests
        unittests_root_dir = teacher_corruption_type.split(":")[1]
        return os.path.join(unittests_root_dir, datapath_constants.DISCRIMINATOR_VALID_TEST_DATA)
    return os.path.join(
        datapath_constants.DISCRIMINATOR_SEQ_VALID_DATA_ROOT,
        teacher_corruption_type,
        f"{dformat}.jsonl",
    )


def _original_task_name(task_name):
    task_name = task_name.replace("seqdialoguediscriminator", "dialogue")
    task_name = task_name.replace("dialoguediscriminator", "dialogue")
    task_name = task_name.replace("humanvsmodeldiscriminator", "dialogue")
    return "_".join(task_name.split("_")[1:])


class BaseDiscriminatorTeacher(BaseDialogueChunkTeacher):
    """
    The base teacher for both REAL and CORRUPTED data streams
    """

    def __init__(self, opt, shared=None) -> None:
        opt = deepcopy(opt)

        # Check incompatible opt
        if self.opt["sequence_discriminator"]:
            assert (
                self.opt["task_version"] == 1
            ), "Code currently only compatible with task version 1"

        super().__init__(opt, shared=shared)

    @property
    def format(self) -> str:
        discriminator_task_name = self.opt["task"].split(":")[0]
        return _original_task_name(discriminator_task_name)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        parser.add_argument(
            "--sequence-discriminator",
            type=bool,
            default=False,
            help="Whether to have simple classifier or a sequence discriminator, which appends the label at the end of message.",
        )
        parser.add_argument(
            "--lowercase-judgement-message",
            type=bool,
            default=False,
            help="If True, the message that is being judged as nonsense/real will be lowercase.",
        )
        return parser

    def flatten_and_maybe_lowercase(self, message: str, phase: Phase, power: Power) -> str:
        messages_dict = self.formatter.messagehistory_unflattener.unflatten_model_output_messages(
            message, power, phase
        )
        assert len(messages_dict) == 1, "Expect only 1 message in message_dict!"

        if self.opt.get("lowercase_judgement_message", False):
            messages_dict[0][MessageObjectPart.MESSAGE] = messages_dict[0][
                MessageObjectPart.MESSAGE
            ].lower()

        flattened_message = self.formatter.messagehistory_flattener.flatten_model_output_messages(messages_dict)  # type: ignore
        return flattened_message


class BaseRealDialogueChunkTeacher(BaseDiscriminatorTeacher):
    """
    Streams the real (no noise) training data with REAL label
    """

    def __init__(self, opt, shared=None) -> None:
        opt = deepcopy(opt)
        self.opt = opt
        super().__init__(opt, shared)
        self.id = "Base Real Dialogue Chunk"

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        for ex in examples:
            # If we are in sequence_discriminator mode we want the label to be of the form
            # [message] [EO_M] [classifier_label]
            # If we are in traditional classifier mode we want the label to be of the form
            # [classifier_label]
            message = ex["labels"][0]
            phase = ex["phase_id"]
            power = ex["player"]
            flatten_message = self.flatten_and_maybe_lowercase(
                message=message, phase=phase, power=power
            )

            if self.opt.get("sequence_discriminator", False):
                ex["text"] = f"{ex['text']}"
                ex["labels"] = [insert_discriminator_label(ex["labels"][0], REAL)]
            else:
                ex["text"] = f"{ex['text']}\n{flatten_message}"
                ex["labels"] = [REAL]
        return examples


@register_teacher("real_message_history_dialoguediscriminator_chunk")
class RealDialogueMessageHistoryChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher("real_message_history_state_dialoguediscriminator_chunk")
class RealDialogueMessageHistoryStateChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher("real_message_history_state_pseudoorder_dialoguediscriminator_chunk")
class RealDialogueMessageHistoryStatePseudoorderChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher("real_message_history_shortstate_pseudoorder_dialoguediscriminator_chunk")
class RealDialogueMessageHistoryShortStatePseudoorderChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher(
    "real_message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_dialoguediscriminator_chunk"
)
@register_teacher("real_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk")
class RealDialogueFullContextChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher(
    "nonsequitur_message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_dialoguediscriminator_chunk"
)
class NonSequiturDialogueFullContextChunkTeacher(BaseRealDialogueChunkTeacher):
    def __init__(self, opt, shared=None) -> None:
        opt = deepcopy(opt)
        opt["remove_n_latest_messages_from_dialogue_history"] = 3
        self.opt = opt
        super().__init__(opt, shared)
        self.id = "Non Sequitur Dialogue Chunk"

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        for ex in examples:
            ex["labels"] = [CORRUPTED]

        return examples


@register_teacher(
    "swapstate_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class SwapStateDialogueFullContextChunkTeacher(BaseRealDialogueChunkTeacher):
    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser = BaseRealDialogueChunkTeacher.add_cmdline_args(argparser, partial_opt)
        argparser.set_defaults(counterfactual_game_cache=500,)
        return argparser

    def __init__(self, opt, shared=None) -> None:
        opt = deepcopy(opt)
        self.opt = opt
        super().__init__(opt, shared)
        self.id = "Corrupt state Dialogue Chunk"

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        for ex in examples:
            ex["labels"] = [CORRUPTED]

        return examples


@register_teacher("real_message_history_shortstate_dialoguediscriminator_chunk")
class RealDialogueMessageHistoryShortStateChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher(
    "real_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class RealDialogueMessageHistoryOrderHistoryChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


@register_teacher(
    "real_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class RealDialogueMessageHistoryOrderHistoryShortStateChunkTeacher(BaseRealDialogueChunkTeacher):
    pass


def change_conversation_participants(parsed_message, sender: bool = False):
    """
    Changes message sender or receiver to some other power

    if `sender` is set to True changes SENDER, otherwise changes RECIPIENT
    """
    if not (
        parsed_message[MessageObjectPart.SENDER] and parsed_message[MessageObjectPart.RECIPIENT]
    ):
        # Probably SILENCE
        return
    party = MessageObjectPart.SENDER if sender else MessageObjectPart.RECIPIENT
    original_power = parsed_message[party]
    other_powers = get_powers([original_power])
    parsed_message[party] = random.choice(other_powers)
    return parsed_message


def change_entities(
    parsed_message, trie, corruption_ratio, corruption_filter,
):
    """
    Changes the name of entities in the text of a message.

    return True if there were any entities to change, otherwise False
    """

    changed = False
    original_message = (
        " " + parsed_message[MessageObjectPart.MESSAGE] + " "
    )  # matching logic needs extra spaces in the start and end of message
    # The line below searches for entities in the `original_message`
    # and returns a dictionary of matches.
    # E.g. if original message is: "Hey England! Should we help France or should we attack france?"
    # then the `matches` result will be:
    # {
    #    ('england', 'powers'): [(4, 13, ' England!')],
    #    ('france', 'powers'): [(28, 36, ' France '), (55, 63, ' france?')]
    # }
    # because we found one instance of "england" and two instances of "france".
    # in different positions.
    matches = change_entity_utils.search_entities(original_message, trie)
    # the key in matches looks like ('england', 'powers'),
    # here 'powers' is the type of entity that has been found in the message.
    # Next, we filter out matches that are not in the entity type defined in `corruption_filter`.
    matches = {
        k: v for k, v in matches.items() if k[1] in corruption_filter
    }  # they key is a tuple of form (entity, entity_type)
    # The line below allows us to sample the ratio of the matches to replace.
    # There are 2 keys in matches for the example:
    # E.g "Hey England! Should we help France or should we attack france?"
    # A corruption_ratio of 1 would keep both matches (for replacement),
    # while a ratio of 0.5 will discard one of the matches.
    # `math.ceil` will ensure at least 1 match will be selected.
    num_replacements = math.ceil(len(matches) * corruption_ratio)
    sampled_matches = random.sample(matches.keys(), num_replacements)  # sample w/o replacement
    # in the loop below, the dictionary `matches` is "flattened" into a single list.
    # for e.g the matches dictionary below:
    # {
    #     ('england', 'powers'): [(4, 13, ' England!')],
    #     ('france', 'powers'): [(28, 36, ' France '), (55, 63, ' france?')]
    # }
    # will be flattened into:
    # [(4, 13, ' England!', ' Italy!'), (28, 36, ' France ', ' Russia '), (55, 63, ' russia?')]
    flattened_sampled_matches = []
    for k in sampled_matches:
        entity, typ = k
        match_instances = matches[k]
        flattened_sampled_matches += change_entity_utils.get_consistant_replacements(
            entity, typ, match_instances
        )

    corrupted_message = original_message[:]
    # We loop over each item in `flattened_sampled_matches` in the reverse order (by start index).
    # For each item , we splice in the replacement into `corrupted_message`.
    for s_idx, e_idx, original_span, repl_span in sorted(
        flattened_sampled_matches, reverse=True
    ):  # swap out from the back...
        corrupted_message = change_entity_utils.splice_replace(
            corrupted_message, s_idx, e_idx, repl_span
        )
        changed = True
    # removing the extra boundary spaces added for search
    corrupted_message = corrupted_message[1:-1]
    if changed:
        parsed_message[MessageObjectPart.MESSAGE] = corrupted_message

    return changed


def pick_message_from_context(game_context):
    """
    Randomly picks a message from context (history)
    """

    def _looks_like_dialogue(msg):
        for symb in (" -> ", ": "):
            if symb not in msg:
                return False
        return True

    context_lines = game_context.split("\n")
    if len(context_lines) < 2:  # Not enough previous context
        return

    candidates = []
    for text_line in context_lines[:-1]:  # throwing out the prompt line
        if not _looks_like_dialogue(text_line):
            continue

        candidates.append(text_line.split(":", 1)[1].strip())

    if candidates:
        return random.choice(candidates)


def message_from_incorrect_phase(parsed_message, game_context):
    """
    Replaces the message with a message from the previous phases of game, if any

    Returns True if there are any messages from previous games to replace with.
    Returning False means that the replacement was not possible.
    """
    selected = pick_message_from_context(game_context)
    if not selected:
        return False
    # Successfully swapped the message with one from previous phases
    parsed_message[MessageObjectPart.MESSAGE] = selected
    return True


###############################################################
#                                                             #
# Corrupted Message Training Teachers                         #
#                                                             #
###############################################################


class BaseCorruptedDialogueChunkTeacher(BaseDiscriminatorTeacher):
    """
    Streams the noisy training data with CORRUPTED label
    """

    def __init__(self, opt, shared=None) -> None:
        opt = deepcopy(opt)
        self.use_corrupted_type_label = opt.get("corrupted_type_label", False)
        self.record_corruption = opt.get("record_corruption_changes", False)
        super().__init__(opt, shared)
        self.id = "Base Corrupted Dialogue Chunk"

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group("Corrupted Dialog Arguments")
        agent.add_argument(
            "--corrupted-type-label",
            type=bool,
            default=False,
            help="Whether to have the name of the corruption class as the calss label.",
        )
        agent.add_argument(
            "--record-corruption-changes",
            type=bool,
            default=True,
            help="Whether to keep a field that indicates how the message was corrupted in the ParlAI Message.",
        )
        return parser

    def _generate_example_tuples(self, game, game_id):
        examples = super()._generate_example_tuples(game, game_id)
        for ex in examples:
            # Some of the noising options are not possible. For example,
            # when we want to replace messages from other phases of the game
            # when there are not such messages.
            # In those cases we return the uncorrupted (REAL) example
            context = ex["text"]
            dialogue_label = ex["labels"][0]
            corrupted_message = self.corrupt_message_str(
                dialogue_label, context, ex["phase_id"], ex["player"]
            )

            if corrupted_message:
                if self.opt.get("sequence_discriminator", False):
                    ex["text"] = f"{context}"
                    ex["labels"] = [
                        insert_discriminator_label(corrupted_message, self.corruption_class())
                    ]
                else:
                    ex["text"] = f"{context}\n{corrupted_message}"
                    ex["labels"] = [self.corruption_class()]
                if self.record_corruption:
                    ex["corruption_changes"] = {
                        "before": dialogue_label,
                        "after": corrupted_message,
                    }
            else:
                message = ex["labels"][0]
                phase = ex["phase_id"]
                power = ex["player"]
                flattened_message = self.flatten_and_maybe_lowercase(message, phase, power)
                if self.opt.get("sequence_discriminator", False):
                    ex["text"] = f"{context}"
                    ex["labels"] = [insert_discriminator_label(dialogue_label, REAL)]
                else:
                    ex["text"] = f"{context}\n{flattened_message}"
                    ex["labels"] = [REAL]

        return examples

    def corrupt_message_str(self, message: str, context: str, phase: Phase, power: Power):
        """
        Corrupts a randomly selected number of messages

        returns corrupted message list if corruption is possible,
        otherwise it returns None
        """
        messages_dict = self.formatter.messagehistory_unflattener.unflatten_model_output_messages(
            message, power, phase
        )

        corrupted = False
        for msg_ind in self.sample_conversations_to_corrupt(len(messages_dict)):
            if self.opt.get("lowercase_judgement_message", False):
                messages_dict[msg_ind][MessageObjectPart.MESSAGE] = messages_dict[msg_ind][
                    MessageObjectPart.MESSAGE
                ].lower()
            corrupted_msg = self.corrupt_message(messages_dict[msg_ind], context)
            if corrupted_msg:
                messages_dict[msg_ind] = corrupted_msg
                corrupted = True

        if corrupted:
            flattened_message = self.formatter.messagehistory_flattener.flatten_model_output_messages(messages_dict)  # type: ignore
            logging.debug(
                "Corruptor teacher changed messages content.\n"
                f"Original: \n{message}\n\nChanged to:\n{flattened_message}"
            )
            return flattened_message
        logging.debug("Corruptor teacher did not change the message.")

    def sample_conversations_to_corrupt(self, num_messages):
        """
        Generates a random list of indices for messages to corrupt.
        """
        if not num_messages:
            return []
        num_samples = random.randint(1, num_messages)
        return random.sample(range(num_messages), k=num_samples)

    def corrupt_message(self, message, context=None):
        """
        Creating a corrupted dialogue message from a real dialogue entry.

        Need to implement this for any class derived from BaseCorruptedDialogueChunkTeacher
        based on the type of noising (message corruption) considered for that class.
        """
        raise NotImplementedError("Message corruption is not implemented for this class")

    def corruption_class(self):
        """
        Returns the corruption class for this teacher
        """
        raise NotImplementedError("Need to specify the corruption class for this teacher.")


class ConversationParticipantNoiseDialogueChunkTeacher(BaseCorruptedDialogueChunkTeacher):
    """
    Streams noisy training data: replaces sender/receiver for other random powers
    """

    def __init__(self, opt, shared=None) -> None:
        super().__init__(opt, shared)
        self.id = "Corrupt Conversation Participants Dialogue Chunk"

    def corruption_class(self):
        return INCORRECT_RECIPIENT if self.use_corrupted_type_label else CORRUPTED

    def corrupt_message(self, message, context):
        if change_conversation_participants(message):
            return message


@register_teacher("corruptedreceiver_message_history_dialoguediscriminator_chunk")
class ConversationParticipantNoiseMessageHistoryChunkTeacher(
    ConversationParticipantNoiseDialogueChunkTeacher
):
    pass


@register_teacher("corruptedreceiver_message_history_state_dialoguediscriminator_chunk")
class ConversationParticipantNoiseMessageHistoryStateChunkTeacher(
    ConversationParticipantNoiseDialogueChunkTeacher
):
    pass


@register_teacher("corruptedreceiver_message_history_shortstate_dialoguediscriminator_chunk")
class ConversationParticipantNoiseMessageHistoryShortStateChunkTeacher(
    ConversationParticipantNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedreceiver_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class ConversationParticipantNoiseMessageHistoryOrderHistoryChunkTeacher(
    ConversationParticipantNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedreceiver_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class ConversationParticipantNoiseMessageHistoryOrderHistoryShortStateChunkTeacher(
    ConversationParticipantNoiseDialogueChunkTeacher
):
    pass


class EntityCorruptedNoiseDialogueChunkTeacher(BaseCorruptedDialogueChunkTeacher):
    """
    Streams noisy training data: detected entities with random other ones
    """

    def __init__(self, opt, shared=None) -> None:
        self.opt = opt
        self.entity_corruption_ratio = opt.get("entity_corruption_ratio", 1.0)
        self.entity_corruption_types = set(opt.get("entity_corruption_types").split(","))
        self.trie = change_entity_utils.build_entity_trie()
        super().__init__(opt, shared)
        self.id = "Corrupt Entities Dialogue Chunk"

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group("Entity Corrupted Dialog Arguments")
        agent.add_argument(
            "--entity-corruption-ratio",
            type=float,
            default=1.0,
            help="What proportion of corruptible entities should actually be corrupted?",
        )
        agent.add_argument(
            "--entity-corruption-types",
            type=str,
            default="powers,power_adjs,locations,symbols,noisy_locations",
            help="comma seperated list of on types of entities should be corrupted (choices: 'powers','power_adjs','locations','symbols','noisy_locations'",
        )
        return parser

    def corruption_class(self):
        return CORRUPTED_ENTITY if self.use_corrupted_type_label else CORRUPTED

    def corrupt_message(self, message, context):
        if change_entities(
            message, self.trie, self.entity_corruption_ratio, self.entity_corruption_types
        ):
            return message


@register_teacher("corruptedentity_message_history_dialoguediscriminator_chunk")
class EntityCorruptedNoiseMessageHistoryChunkTeacher(EntityCorruptedNoiseDialogueChunkTeacher):
    pass


@register_teacher("corruptedentity_message_history_state_dialoguediscriminator_chunk")
class EntityCorruptedNoiseMessageHistoryStateChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher("corruptedentity_message_history_state_pseudoorder_dialoguediscriminator_chunk")
class EntityCorruptedNoiseMessageHistoryPseudoorderStateChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedentity_message_history_shortstate_pseudoorder_dialoguediscriminator_chunk"
)
class EntityCorruptedNoiseMessageHistoryShortstatePseudoorderChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedentity_message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_dialoguediscriminator_chunk"
)
class EntityCorruptedNoiseFullContextChunkTeacher(EntityCorruptedNoiseDialogueChunkTeacher):
    pass


@register_teacher("corruptedentity_message_history_shortstate_dialoguediscriminator_chunk")
class EntityCorruptedNoiseMessageHistoryShortStateChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedentity_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class EntityCorruptedNoiseMessageHistoryOrderHistoryChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "corruptedentity_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class EntityCorruptedNoiseMessageHistoryOrderHistoryShortStateChunkTeacher(
    EntityCorruptedNoiseDialogueChunkTeacher
):
    pass


class IncorrectPhaseNoiseDialogueChunkTeacher(BaseCorruptedDialogueChunkTeacher):
    """
    Streams noisy training data: message is swapped from a message from previous phases
    """

    def __init__(self, opt, shared=None) -> None:
        super().__init__(opt, shared)
        self.id = "Incorrect Phase Message Dialogue Chunk"

    def corruption_class(self):
        return INCORRECT_PHASE if self.use_corrupted_type_label else CORRUPTED

    def corrupt_message(self, message, context):
        if message_from_incorrect_phase(message, context):
            return message


@register_teacher("incorrectphase_message_history_dialoguediscriminator_chunk")
class IncorrectPhaseNoiseMessageHistoryChunkTeacher(IncorrectPhaseNoiseDialogueChunkTeacher):
    pass


@register_teacher("incorrectphase_message_history_state_dialoguediscriminator_chunk")
class IncorrectPhaseNoiseMessageHistoryStateChunkTeacher(IncorrectPhaseNoiseDialogueChunkTeacher):
    pass


@register_teacher("incorrectphase_message_history_shortstate_dialoguediscriminator_chunk")
class IncorrectPhaseNoiseMessageHistoryShortStateChunkTeacher(
    IncorrectPhaseNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "incorrectphase_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class IncorrectPhaseNoiseMessageHistoryOrderHistoryChunkTeacher(
    IncorrectPhaseNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "incorrectphase_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class IncorrectPhaseNoiseMessageHistoryOrderHistoryShortStateChunkTeacher(
    IncorrectPhaseNoiseDialogueChunkTeacher
):
    pass


class IncorrectGameCorruptedNoiseDialogueChunkTeacher(BaseCorruptedDialogueChunkTeacher):
    """
    Streams noisy training data: message is swapped from a message from a different example/game
    """

    def __init__(self, opt, shared=None) -> None:
        self._previous_msgs_cache = []
        super().__init__(opt, shared)
        self.id = "Incorrect Game Message Dialogue Chunk"

    def get_corrupted_message(self):
        self._check_shuffle_cut()
        return random.choice(self._previous_msgs_cache)

    def _check_shuffle_cut(self):
        if len(self._previous_msgs_cache) > GAME_CACHE_BUFFER_SIZE:
            random.shuffle(self._previous_msgs_cache)
            new_size = GAME_CACHE_BUFFER_SIZE // 2
            self._previous_msgs_cache = self._previous_msgs_cache[:new_size]

    def _add_to_cache(self, message):
        self._previous_msgs_cache.append(message[MessageObjectPart.MESSAGE])

    def corruption_class(self):
        return INCORRECT_GAME if self.use_corrupted_type_label else CORRUPTED

    def corrupt_message(self, message, context):
        # cache of previous games is not started yet
        if not self._previous_msgs_cache:
            self._add_to_cache(message)
            return

        corrupted_message = deepcopy(message)
        corrupted_message[MessageObjectPart.MESSAGE] = self.get_corrupted_message()
        self._add_to_cache(message)
        return corrupted_message


@register_teacher("incorrectgame_message_history_dialoguediscriminator_chunk")
class IncorrectGameNoiseMessageHistoryChunkTeacher(
    IncorrectGameCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher("incorrectgame_message_history_state_dialoguediscriminator_chunk")
class IncorrectGameNoiseMessageHistoryStateChunkTeacher(
    IncorrectGameCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher("incorrectgame_message_history_shortstate_dialoguediscriminator_chunk")
class IncorrectGameNoiseMessageHistoryShortStateChunkTeacher(
    IncorrectGameCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "incorrectgame_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class IncorrectGameNoiseMessageHistoryOrderHistoryChunkTeacher(
    IncorrectGameCorruptedNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "incorrectgame_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class IncorrectGameNoiseMessageHistoryOrderHistoryShortStateChunkTeacher(
    IncorrectGameCorruptedNoiseDialogueChunkTeacher
):
    pass


class RepeatMessageNoiseDialogueChunkTeacher(BaseCorruptedDialogueChunkTeacher):
    """
    Streams noisy training data: one of the messages are repeated twice
    """

    def __init__(self, opt, shared=None) -> None:
        super().__init__(opt, shared)
        self.id = "Repeat Message Dialogue Chunk"

    def corruption_class(self):
        return REAPEATED_MESSAGE if self.use_corrupted_type_label else CORRUPTED

    def corrupt_message_str(self, message: str, context: str, phase: Phase, power: Power):
        """
        Corrupts a randomly selected number of messages

        returns corrupted message list if corruption is possible,
        otherwise it returns None
        """
        messages_dict = self.formatter.messagehistory_unflattener.unflatten_model_output_messages(
            message, power, phase
        )
        if not messages_dict:
            logging.debug("No message to repeat. Skipping message corruption.")
            return

        repeated_message = random.choice(messages_dict)
        messages_dict.append(repeated_message)
        random.shuffle(messages_dict)

        flattened_message = self.formatter.messagehistory_flattener.flatten_model_output_messages(messages_dict)  # type: ignore

        logging.debug(
            "Corruptor teacher changed messages content.\n"
            f"Original: \n{message}\n\nChanged to:\n{flattened_message}"
        )
        return flattened_message


@register_teacher("repeatmessage_message_history_dialoguediscriminator_chunk")
class RepeatMessageNoiseMessageHistoryChunkTeacher(RepeatMessageNoiseDialogueChunkTeacher):
    pass


@register_teacher("repeatmessage_message_history_state_dialoguediscriminator_chunk")
class RepeatMessageNoiseMessageHistoryStateChunkTeacher(RepeatMessageNoiseDialogueChunkTeacher):
    pass


@register_teacher("repeatmessage_message_history_shortstate_dialoguediscriminator_chunk")
class RepeatMessageNoiseMessageHistoryShortStateChunkTeacher(
    RepeatMessageNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "repeatmessage_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_chunk"
)
class RepeatMessageNoiseMessageHistoryOrderHistoryChunkTeacher(
    RepeatMessageNoiseDialogueChunkTeacher
):
    pass


@register_teacher(
    "repeatmessage_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_chunk"
)
class RepeatMessageNoiseMessageHistoryOrderHistoryShortStateChunkTeacher(
    RepeatMessageNoiseDialogueChunkTeacher
):
    pass


###############################################################
#                                                             #
# Validation Teachers                                         #
#                                                             #
###############################################################

CORRUPTED_VALID_DATA_DUMP_TYPES = (
    "corruptedreceiver",
    "corruptedentity",
    "incorrectphase",
    "incorrectgame",
    "repeatmessage",
)
REAL_VALID_DATA_DUMP = "real"


class BaseDiscriminatorValidationTeacher(DialogTeacher):
    """
    Base class for streaming noisy and real validation data to train discriminative classifier
    """

    def __init__(self, opt, shared=None):
        self.opt = deepcopy(opt)
        self.examples_per_corr_type = min(
            self.opt.get("n_valid_examples_per_type", DEFAULT_MAX_VALID_EXAMPLE_PER_TYPE),
            MAX_AVAILABLE_VALID_EXAMPLES_PER_TYPE,
        )
        self.use_corrupted_type_label = self.opt.get("corrupted_type_label", False)
        assert (
            "datafile" in self.opt
            and isinstance(self.opt["datafile"], tuple)
            and len(self.opt["datafile"])
        ), f"Invalid or no datafile found in opt. Found: {self.opt['datafile']}"
        super().__init__(self.opt, shared=shared)  # type: ignore
        self.id = "Base Validation Data Teacher"

    def _get_format_from_task_name(self):
        task_name_parts = self.opt["task"].split("_")
        return "_".join(task_name_parts[1:-2])

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group("Discriminator Eval Teachers Arguments")
        agent.add_argument(
            "--corrupted-type-label",
            type=bool,
            default=False,
            help="Whether to have the name of the corruption class as the calss label.",
        )
        agent.add_argument(
            "--n-valid-examples-per-type",
            type=int,
            default=DEFAULT_MAX_VALID_EXAMPLE_PER_TYPE,
            help=(
                "The maximum number of examples to load for each requested corrupted/real type."
            ),
        )
        agent.add_argument(
            "--sequence-discriminator",
            type=bool,
            default=False,
            help="Whether to have simple classifier or a sequence discriminator, which appends the label at the end of message.",
        )
        return parser

    def num_examples(self):
        return self.examples_per_corr_type * len(self.opt["datafile"])

    def num_episodes(self):
        return self.num_examples()

    def get_label(self, example):
        label = example.pop("eval_labels")[0]
        if label == REAL or self.use_corrupted_type_label:
            # Returning the assigned label (real, or the corruption type)
            return [label]
        return [CORRUPTED]

    def setup_data(self, path):
        if type(path) is not tuple:
            path = (path,)
        data = []
        for dump_path in path:
            if not dump_path:
                continue
            logging.info(f"Loading data from {dump_path}")
            with jsonlines.open(dump_path, "r") as fi:
                data.extend([js for i, js in enumerate(fi) if i < self.examples_per_corr_type])

        for valid_data in data:
            example = valid_data["dialog"][0][0]
            episode_done = example.pop("episode_done")
            example["labels"] = self.get_label(example)
            yield example, episode_done


class BaseCorruptedMessagesValidTeacher(BaseDiscriminatorValidationTeacher):
    """
    Corrupted validation data teacher
    """

    def __init__(self, opt, shared=None):
        self.opt = deepcopy(opt)
        self._format = self._get_format_from_task_name()
        self.opt["datafile"] = self._collect_datafiles(opt.get("valid_corrupted_dtype", ""))
        super().__init__(self.opt, shared=shared)
        self.id = "Corrupted Validation Data Teacher"

    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt=None) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group("Discriminator Task Corrupted Eval Data Teacher")
        agent.add_argument(
            "--valid-corrupted-dtype",
            type=str,
            default="",
            help=(
                "Comma-seperated list of corrupted data dumps to use. "
                "Default (empty string) is all of them."
            ),
        )
        agent.add_argument(
            "--n-corrupted-valid-examples",
            type=int,
            default=DEFAULT_MAX_VALID_EXAMPLE_PER_TYPE,
            help=(
                "The maximum number of corrupted valid examples to load from each validation data dump."
            ),
        )

    def _collect_datafiles(self, valid_corrptd_types):
        if valid_corrptd_types:
            requested_types = valid_corrptd_types.split(",")
        else:
            requested_types = CORRUPTED_VALID_DATA_DUMP_TYPES

        datafiles = []
        for dtype in requested_types:
            if self.opt.get("sequence_discriminator", False):
                dpath = _get_seq_valid_dpath(dtype, self._format)
            else:
                dpath = _get_valid_dpath(dtype, self._format)
            assert os.path.isfile(dpath), (
                f"Data source file not found in {dpath}. "
                f'Maybe invalid requested data dumps type "{dtype}". '
                f"Valid corrupted types are {CORRUPTED_VALID_DATA_DUMP_TYPES}"
            )
            datafiles.append(dpath)

        return tuple(datafiles)


@register_teacher("validcorrupted_message_history_dialoguediscriminator_evaldump")
class CorruptedMessageHistoryValidTeacher(BaseCorruptedMessagesValidTeacher):
    pass


@register_teacher("validcorrupted_message_history_state_dialoguediscriminator_evaldump")
class CorruptedMessageHistoryStateValidTeacher(BaseCorruptedMessagesValidTeacher):
    pass


@register_teacher("validcorrupted_message_history_shortstate_dialoguediscriminator_evaldump")
class CorruptedMessageHistoryShortStateValidTeacher(BaseCorruptedMessagesValidTeacher):
    pass


@register_teacher(
    "validcorrupted_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_evaldump"
)
class CorruptedMessageHistoryOrderHistoryValidTeacher(BaseCorruptedMessagesValidTeacher):
    pass


@register_teacher(
    "validcorrupted_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_evaldump"
)
class CorruptedMessageHistoryOrderHistoryShortStateValidTeacher(BaseCorruptedMessagesValidTeacher):
    pass


class BaseRealMessagesValidTeacher(BaseDiscriminatorValidationTeacher):
    """
    Real validation data teacher
    """

    def __init__(self, opt, shared=None):
        self.opt = deepcopy(opt)
        self._format = self._get_format_from_task_name()
        if self.opt.get("sequence_discriminator", False):
            self.opt["datafile"] = tuple(
                [_get_seq_valid_dpath(REAL_VALID_DATA_DUMP, self._format)]
            )
        else:
            self.opt["datafile"] = tuple([_get_valid_dpath(REAL_VALID_DATA_DUMP, self._format)])
        super().__init__(self.opt, shared=shared)
        self.id = "Real Validation Data Teacher"


@register_teacher("validreal_message_history_dialoguediscriminator_evaldump")
class RealMessageHistoryValidTeacher(BaseRealMessagesValidTeacher):
    pass


@register_teacher("validreal_message_history_state_dialoguediscriminator_evaldump")
class RealMessageHistoryStateValidTeacher(BaseRealMessagesValidTeacher):
    pass


@register_teacher("validreal_message_history_shortstate_dialoguediscriminator_evaldump")
class RealMessageHistoryShortStateValidTeacher(BaseRealMessagesValidTeacher):
    pass


@register_teacher(
    "validreal_message_history_orderhistorysincelastmovementphase_shortstate_dialoguediscriminator_evaldump"
)
class RealMessageHistoryOrderHistoryShortStateValidTeacher(BaseRealMessagesValidTeacher):
    pass


@register_teacher(
    "validreal_message_history_orderhistorysincelastmovementphase_dialoguediscriminator_evaldump"
)
class RealMessageHistoryOrderHistoryStateValidTeacher(BaseRealMessagesValidTeacher):
    pass


class HumanVsModelDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Streaming data base dialogue teacher for messages/orders.

    Loads predicted pseudo orders

    Label is next message
    """

    @staticmethod
    def add_cmdline_args(argparser, partial_opt=None):
        argparser.add_argument(
            "--model-generated-messages",
            type=str,
            default="nucleus_0.9",
            choices=datapath_constants.GENERATED_MESSAGES_DIRS.keys(),
            help="What kind of model generated messages should the classifier train against?",
        )
        argparser.add_argument(
            "--blend-generations",
            type=bool,
            default=False,
            help="Blend generations on conjunctions and punctuation",
        )
        argparser = BaseDialogueChunkTeacher.add_cmdline_args(argparser, partial_opt)
        return argparser

    def __init__(self, opt, shared=None):
        self.opt = opt

        # check incompatible opt
        assert not self.opt["add_sleep_times"], "Sleep times incompatible with discriminator"

        self._set_model_generated_messages_dirs()

        super().__init__(opt, shared)
        self.id = "Base Dialogue Chunk with pseudo orders and model generated messages"

    def _set_model_generated_messages_dirs(self):
        model_option = self.opt.get("model_generated_messages", "nucleus_0.9")
        if model_option in datapath_constants.GENERATED_MESSAGES_DIRS:
            self.model_generated_messages_dirs = datapath_constants.GENERATED_MESSAGES_DIRS[
                model_option
            ]
        else:
            raise ValueError(f"{model_option} is not a known model generated message type")

        logging.info(f"Model generated messages location {self.model_generated_messages_dirs}")

    def _load_model_generated_messages(self, game_id: int):
        paths = [
            os.path.join(generated_messages_dir, f"game_{game_id}_model_dialogue.json")
            for generated_messages_dir in self.model_generated_messages_dirs
        ]

        for path in paths:
            if not os.path.isfile(path):
                # logging.error(
                #    f"Model generated messages missing for game ID: {game_id}, expected file {path}"
                # )
                return []

        logging.error(f"Loading generated messages from following paths: {paths}")
        return [load_json(path) for path in paths]

    def get_player_metadata(self, game, game_id):
        metadata = super().get_player_metadata(game, game_id)
        metadata["model_generated_messages"] = self._load_model_generated_messages(game_id)
        return metadata


@register_teacher("message_history_pseudoorder_humanvsmodeldiscriminator_chunk")
class MessageHistoryPseudoorderHumanVsModelDialogueChunkTeacher(HumanVsModelDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_humanvsmodeldiscriminator_chunk")
class MessageHistoryHumanVsModelDialogueChunkTeacher(HumanVsModelDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_shortstate_humanvsmodeldiscriminator_chunk")
class MessageHistoryShortStateHumanVsModelDialogueChunkTeacher(HumanVsModelDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    pass


@register_teacher("message_history_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk")
class MessageHistoryShortStatePseudoorderHumanVsModelDialogueChunkTeacher(
    HumanVsModelDialogueChunkTeacher
):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    pass


@register_teacher(
    "message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk"
)
class MessageHistoryFullContextHumanVsModelDialogueChunkTeacher(HumanVsModelDialogueChunkTeacher):
    pass


@register_teacher(
    "orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk"
)
class NoMessageHistoryFullContextHumanVsModelDialogueChunkTeacher(
    HumanVsModelDialogueChunkTeacher
):
    pass
