#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.typedefs import (
    Location,
    Order,
    Power,
    JointAction,
    Action,
    Phase,
    Timestamp,
    MessageDict,
)

import typing
import torch
import numpy

class PhaseData:
    def to_dict(self) -> dict: ...
    @property
    def messages(self) -> typing.Dict[Timestamp, MessageDict]: ...
    @property
    def name(self) -> Phase: ...
    @property
    def orders(self) -> typing.Dict[Power, Action]: ...
    @property
    def state(self) -> typing.Dict[str, typing.Dict[Power, typing.Any]]: ...
    def get_scores(self, scoring_system: int) -> typing.List[float]: ...

class Game:
    @typing.overload
    def __init__(self, arg0: Game) -> None: ...
    @typing.overload
    def __init__(self, draw_on_stalemate_years: int = -1, is_full_press: bool = True) -> None: ...
    def add_log(self, arg0: str) -> None: ...
    def add_message(
        self,
        sender: Power,
        recipient: Power,
        body: str,
        time_sent: Timestamp,
        increment_on_collision: bool = False,
    ) -> None: ...
    def clear_old_all_possible_orders(self) -> None: ...
    def clear_orders(self) -> None: ...
    def compute_board_hash(self) -> int: ...
    def compute_order_history_hash(self) -> int: ...
    @classmethod
    def from_json(cls, str) -> Game: ...
    def from_json_inplace(self, Game) -> None: ...
    def get_alive_power_ids(self) -> typing.List[int]: ...
    def get_alive_powers(self) -> typing.List[str]: ...
    def get_all_phase_names(self) -> typing.List[str]: ...
    def get_all_phases(self) -> typing.List[PhaseData]:
        """
        Gets the phase data for all past phases and the current staged phase.
        """
    def get_all_possible_orders(self) -> typing.Dict[Location, typing.List[Order]]: ...
    def get_current_phase(self) -> str: ...
    def get_logs(self) -> dict: ...
    def get_metadata(self, k: str) -> str: ...
    def get_next_phase(self, phase: str) -> typing.Optional[str]: ...
    def get_orderable_locations(self) -> typing.Dict[Power, typing.List[Location]]: ...
    def get_orders(self) -> typing.Dict[Power, typing.List[Order]]: ...
    def get_phase_data(self) -> PhaseData:
        """
        NOTE: get_phase_data, bizarrely, omits the staged orders and messagesof the current phase. This can lead to unexpected bugs, for exampleattempting to walk through the phase-by-phase messages of a game by playingthrough get_phase_history and then get_phase_data will NOT find all messages.Use get_all_phases or get_staged_phase_data, which do not have this behavior.
        """
    def get_phase_history(self) -> typing.List[PhaseData]:
        """
        Gets the phase data for all past phases, not including the current staged phase.
        """
    def get_prev_phase(self, phase: str) -> typing.Optional[str]: ...
    SCORING_SOS: int
    """sum of squares. In case of non-solo, score is proportional to centers^2"""
    SCORING_DSS: int
    """draw size scoring. In case of non-solo, score is equal among surviving players"""
    def get_scores(self, scoring_system: int = SCORING_SOS) -> typing.List[float]: ...
    def get_scoring_system(self) -> int: ...
    def set_scoring_system(self, scoring_system: int) -> None: ...
    def get_staged_phase_data(self) -> PhaseData:
        """
        Gets the phase data for the current staged phase that is not processed yet.
        """
    def get_state(self) -> dict: ...
    def get_units(self) -> dict[Power, typing.List[str]]: ...
    def get_unit_power_at(self, loc: str) -> typing.Optional[str]:
        """Returns the owning power of a unit there if it exists, returns None if no unit there.
        If a location is specified without coasts, then tests that location and all possible
        coastal extensions of it. If a location is specified with a coast, only finds the unit if
        it is there at that exact coast.
        """
    def get_unit_type_at(self, loc: str) -> typing.Optional[str]:
        """Returns the type a unit there if it exists, returns None if no unit there.
        If a location is specified without coasts, then tests that location and all possible
        coastal extensions of it. If a location is specified with a coast, only finds the unit if
        it is there at that exact coast.
        """
    def is_supply_center(self, loc: str) -> bool:
        """Returns true if the location is an SC. Locations specified with coasts are also valid."""
    def get_supply_center_power(self, loc: str) -> typing.Optional[str]:
        """Returns the current owner of the SC if the location is an SC and if the SC is owned
        by some power. Locations specified with coasts are also valid."""
    def phase_of_last_message_at_or_before(self, timestamp: Timestamp) -> Phase: ...
    def process(self) -> None: ...
    def rollback_messages_to_timestamp_start(self, timestamp: Timestamp) -> None: ...
    def rollback_messages_to_timestamp_end(self, timestamp: Timestamp) -> None: ...
    def delete_message_at_timestamp(self, timestamp: Timestamp) -> None: ...
    def get_last_message_timestamp(self) -> Timestamp: ...
    def rolled_back_to_phase_end(self, phase: str) -> Game: ...
    def rolled_back_to_phase_start(self, phase: str) -> Game: ...
    def rolled_back_to_timestamp_end(self, timestamp: Timestamp) -> Game: ...
    def rolled_back_to_timestamp_start(self, timestamp: Timestamp) -> Game: ...
    def set_all_orders(self, arg0: JointAction) -> None:
        """
        NOTE: Clears and replaces any existing staged orders for any power
        """
    def get_consecutive_years_without_sc_change(self) -> int: ...
    def any_sc_occupied_by_new_power(self) -> bool: ...
    def set_draw_on_stalemate_years(self, arg0: int) -> None: ...
    def set_exception_on_convoy_paradox(self) -> None: ...
    def set_metadata(self, k: str, v: str) -> None: ...
    def set_orders(self, arg0: Power, arg1: typing.Sequence[Order]) -> None: ...
    def to_json(self) -> str: ...
    def clone_n_times(self, n_repeats: int) -> typing.Sequence["Game"]: ...
    @property
    def current_short_phase(self) -> Phase: ...
    @property
    def current_year(self) -> int: ...
    @property
    def game_id(self) -> str:
        """
        :type: str
        """
    @game_id.setter
    def game_id(self, arg0: str) -> None:
        pass
    @property
    def is_full_press(self) -> bool: ...
    @property
    def is_game_done(self) -> bool:
        """
        :type: bool
        """
    @property
    def map_name(self) -> str:
        """
        :type: str
        """
    @property
    def message_history(self) -> typing.Dict[Phase, typing.Dict[Timestamp, MessageDict]]:
        """
        Message history for **previous** phase. Use game.get_all_phases to get all messages.
        """
    @property
    def messages(self) -> typing.Dict[Timestamp, MessageDict]:
        """
        :type: dict
        """
    @property
    def phase(self) -> str:
        """
        :type: str
        """
    @property
    def phase_type(self) -> str:
        """
        :type: str
        """
    LOC_STRS: typing.List[str]
    @classmethod
    def is_water(cls, loc: str) -> bool: ...
    @classmethod
    def is_coast(cls, loc: str) -> bool: ...
    @classmethod
    def is_center(cls, loc: str) -> bool: ...

class SinglePowerCFRStats:
    def __init__(
        self,
        use_linear_weighting: bool,
        use_optimistic_cfr: bool,
        qre: bool,
        qre_target_blueprint: bool,
        qre_eta: float,
        qre_lambda: float,
        qre_entropy_factor: float,
        bp_action_relprobs: typing.List[float],
    ):
        """
        Arguments:
        use_linear_weighting: Weight iteration t by t, instead of uniformly.
        cfr_optimistic: Only matters if not qre, the last iteration counts double.
        qre: If true, use qre, else use cfr
        qre_target_blueprint: Only matters if qre. Bias towards
          bp_prob instead of uniform.
        qre_eta: Only matters if qre. Parameter that controls
          convergence of qre.
        qre_lambda: lambda in qre. Strength of bias towards uniform
          or blueprint.
        qre_entropy_factor: KL reg consists of agent minimizing
          sum_a agent(a) log(target(a)) - agent(a) log(agent(a))
          First term is cross entropy, second term is entropy.
          This factor scales the entropy term.
        bp_action_relprobs: the vector of blueprint probabilities of
          the plausible actions. All further functions in this class
          that deal with vectors of per-action values will adhere to
          the same ordering.
        """
    ACCUMULATE_PREV_ITER: int
    """Pass this to update as which_strategy_to_accumulate to accumulate the previous iteration strategy itno the average strategy"""
    ACCUMULATE_BLUEPRINT: int
    """Pass this to update as which_strategy_to_accumulate to accumulate the blueprint into the average strategy"""
    def update(
        self,
        state_utility: float,
        action_utilities: typing.List[float],
        which_strategy_to_accumulate: int,
        cfr_iter: int,
    ):
        """
        Update stats after an iteration.
        Arguments:
        state_utility: the actual utility achieved on this iteration
        action_utilities: the utility for each action for this player
        which_strategy_to_accumulate: one of ACCUMULATE_PREV_ITER or
          ACCUMULATE_BLUEPRINT.
        cfr_iter: the 0-indexed iteration of CFR just finished.
        """
    def cur_iter_strategy(self) -> typing.List[float]: ...
    def bp_strategy(self, temperature: float) -> typing.List[float]: ...
    def avg_strategy(self) -> typing.List[float]: ...
    def avg_action_utilities(self) -> typing.List[float]: ...
    def cur_iter_action_prob(self, action_idx: int) -> float: ...
    def avg_action_prob(self, action_idx: int) -> float: ...
    def avg_action_utility(self, action_idx: int) -> float: ...
    def avg_action_regret(self, action_idx: int) -> float: ...
    def avg_utility(self) -> float: ...
    def avg_utility_stdev(self) -> float: ...
    def __getstate__(self) -> typing.Any: ...
    def __setstate__(self, d: typing.Any): ...

class CFRStats:
    def __init__(
        self,
        use_linear_weighting: bool,
        cfr_optimistic: bool,
        qre: bool,
        qre_target_blueprint: bool,
        qre_eta: float,
        power_qre_lambda: typing.Dict[Power, float],
        power_qre_entropy_factor: typing.Dict[Power, float],
        bp_action_relprobs_by_power: typing.Dict[str, typing.List[float]],
    ):
        """
        Arguments:
        use_linear_weighting: Weight iteration t by t, instead of uniformly.
        cfr_optimistic: Only matters if not qre, the last iteration counts double.
        qre: If true, use qre, else use cfr
        qre_target_blueprint: Only matters if qre. Bias towards bp_prob instead of
          uniform.
        qre_eta: Only matters if qre. Parameter that controls convergence
          of qre.
        power_qre_lambda: Mapping from power to lambda.
          Only matters if qre. Strength of bias towards uniform
          or blueprint.
        power_qre_entropy_factor: KL reg consists of agent minimizing
          sum_a agent(a) log(target(a)) - agent(a) log(agent(a))
          First term is cross entropy, second term is entropy.
          This factor scales the entropy term.
        bp_action_relprobs_by_power: For each power, the vector of
          blueprint probabilities of the plausible actions for that power.
          All further functions in this class that deal with vectors of
          per-action values will adhere to the same ordering.
        """
    ACCUMULATE_PREV_ITER: int
    """Pass this to update as which_strategy_to_accumulate to accumulate the previous iteration strategy itno the average strategy"""
    ACCUMULATE_BLUEPRINT: int
    """Pass this to update as which_strategy_to_accumulate to accumulate the blueprint into the average strategy"""
    def update(
        self,
        power: Power,
        state_utility: float,
        action_utilities: typing.List[float],
        which_strategy_to_accumulate: int,
        cfr_iter: int,
    ):
        """
        Update stats for a given power after an iteration.
        Arguments:
        power: the power to update
        state_utility: the actual utility achieved on this iteration
        action_utilities: the utility for each action for this player
        which_strategy_to_accumulate: one of ACCUMULATE_PREV_ITER or
          ACCUMULATE_BLUEPRINT.
        cfr_iter: the 0-indexed iteration of CFR just finished.
        """
    def cur_iter_strategy(self, power: Power) -> typing.List[float]: ...
    def bp_strategy(self, power: Power, temperature: float) -> typing.List[float]: ...
    def avg_strategy(self, power: Power) -> typing.List[float]: ...
    def avg_action_utilities(self, power: Power) -> typing.List[float]: ...
    def cur_iter_action_prob(self, power: Power, action_idx: int) -> float: ...
    def avg_action_prob(self, power: Power, action_idx: int) -> float: ...
    def avg_action_utility(self, power: Power, action_idx: int) -> float: ...
    def avg_action_regret(self, power: Power, action_idx: int) -> float: ...
    def avg_utility(self, power: Power) -> float: ...
    def avg_utility_stdev(self, power: Power) -> float: ...
    def __getstate__(self) -> typing.Any: ...
    def __setstate__(self, d: typing.Any): ...

class ThreadPool:
    def __init__(self, arg0: int, arg1: typing.Dict[str, int], arg2: int) -> None: ...
    def decode_order_idxs(
        self, order_idxs: torch.Tensor
    ) -> typing.List[typing.List[typing.List[Order]]]: ...
    def decode_order_idxs_all_powers(
        self,
        order_idxs: torch.Tensor,
        x_in_adj_phase: torch.Tensor,
        x_power: torch.Tensor,
        batch_repeat_interleave: int,
    ) -> typing.List[typing.List[typing.List[Order]]]: ...
    def encode_orders_single_strict(
        self, arg0: typing.Sequence[str], arg1: int
    ) -> torch.Tensor: ...
    def encode_orders_single_tolerant(
        self, arg0: Game, arg1: typing.Sequence[str], arg2: int
    ) -> torch.Tensor: ...
    def encode_inputs_all_powers_multi(
        self, arg0: typing.Sequence[Game], arg1: int
    ) -> typing.Dict[str, torch.Tensor]: ...
    def encode_inputs_multi(
        self, arg0: typing.Sequence[Game], arg1: int
    ) -> typing.Dict[str, torch.Tensor]: ...
    def encode_inputs_state_only_multi(
        self, arg0: typing.Sequence[Game], arg1: int
    ) -> typing.Dict[str, torch.Tensor]: ...
    def process_multi(self, arg0: typing.Sequence[Game]) -> None: ...

def encode_board_state(*args, **kwargs) -> typing.Any:
    pass

def encode_board_state_from_json(arg0: str, arg1: int) -> numpy.ndarray[numpy.float32]:
    pass

def encode_board_state_from_phase(arg0: PhaseData, arg1: int) -> numpy.ndarray[numpy.float32]:
    pass

def encode_board_state_pperm_matrices(
    arg0: numpy.ndarray[numpy.int32], arg1: int
) -> numpy.ndarray[numpy.float32]:
    pass

def encoding_sc_ownership_idxs(arg0: int) -> typing.List[int]:
    pass

def encoding_unit_ownership_idxs(arg0: int) -> typing.List[int]:
    pass

def max_input_version() -> int:
    """Return the current latest input_version supported by the board/order encoder."""
    ...

def board_state_enc_width(input_version: int) -> int:
    """Return number of features in board encoding for this input_version."""
    ...
