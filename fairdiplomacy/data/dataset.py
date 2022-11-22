#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, Union, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional
from tqdm import tqdm
from fairdiplomacy.models.base_strategy_model.base_strategy_model import Scoring
from fairdiplomacy.agents.base_agent import NoAgentState

from conf import conf_cfgs
from fairdiplomacy.data.data_fields import DataFields
from fairdiplomacy.models.consts import POWERS, MAX_SEQ_LEN, LOCS, N_SCS
from fairdiplomacy.models.state_space import EOS_IDX
from fairdiplomacy.models.base_strategy_model.util import (
    explain_base_strategy_model_decoder_inputs,
)
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Action, JointAction, Phase, Power
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.game import year_of_phase
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state
from fairdiplomacy.utils.game_id import extract_game_id_str, extract_game_id_int
from fairdiplomacy.utils.order_idxs import (
    action_strs_to_global_idxs,
    global_order_idxs_to_str,
    MAX_VALID_LEN,
    OrderIdxConversionException,
    local_order_idxs_to_global,
)
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.tensorlist import TensorList
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder, DEFAULT_INPUT_VERSION

LOC_IDX = {loc: idx for idx, loc in enumerate(LOCS)}


class Dataset(torch.utils.data.Dataset):  # type:ignore
    def __init__(
        self,
        cfg: conf_cfgs.NoPressDatasetParams,
        *,
        use_validation: bool,
        all_powers=False,
        input_version: int = DEFAULT_INPUT_VERSION,
        max_year: Optional[int] = None,
    ):
        torch.set_num_threads(1)

        self.debug_only_opening_phase = cfg.debug_only_opening_phase
        self.exclude_n_holds = cfg.exclude_n_holds
        self.n_cf_agent_samples = cfg.n_cf_agent_samples
        self.only_with_min_final_score = cfg.only_with_min_final_score
        # only use value_dir for train set, not val set. We still want to validate on *real* values!
        self.value_dir = None if use_validation else cfg.value_dir
        self.limit_n_games = cfg.limit_n_games
        self.all_powers = all_powers
        self.input_version = input_version
        self.return_hold_for_invalid = cfg.return_hold_for_invalid
        self.max_year = max_year

        game_data_path = cfg.val_set_path if use_validation else cfg.train_set_path
        assert game_data_path is not None
        # Allow no metadata when just doing inference-time validation testing
        # on different datasets
        assert cfg.metadata_path is not None or use_validation

        self.cf_agent = None

        self.game_metadata = None
        if cfg.metadata_path is not None:
            logging.info(f"Reading metadata from {cfg.metadata_path}")
            with open(cfg.metadata_path) as meta_f:
                self.game_metadata = json.load(meta_f)
        else:
            logging.info(f"Metadata path is None, skipping it")

        # Metadata keys are sometimes paths, sometimes int. Be consistent here.
        extract_game_id_fn = (
            extract_game_id_int
            if self.game_metadata is not None
            and is_int(next(k for k in self.game_metadata.keys()))
            else extract_game_id_str
        )
        if self.game_metadata is not None:
            self.game_metadata = {extract_game_id_fn(k): v for k, v in self.game_metadata.items()}

        self.min_rating_percentile = cfg.min_rating_percentile
        self.min_total_games = cfg.min_total_games

        logging.info(
            f"Only training on powers with min rating percentile {self.min_rating_percentile} and min games {self.min_total_games}"
        )
        if self.game_metadata is not None:
            add_rating_percentiles_to_metadata(self.game_metadata)

        paths_and_jsons = get_paths_and_jsons(game_data_path, cfg.limit_n_games)

        if self.limit_n_games > 0:
            logging.info(f"Using only first {self.limit_n_games} games")
            n_games = self.limit_n_games
        else:
            logging.info("Skimming games data to get # games")
            n_games = len(paths_and_jsons)
            logging.info(f"Found {n_games} games")

        def read_data(paths_and_jsons):
            for game_path, game_json in paths_and_jsons:
                if self.game_metadata is None:
                    yield game_path, game_json

                elif game_path in self.game_metadata:
                    yield game_path, game_json
                else:
                    game_id = extract_game_id_fn(game_path)
                    if game_id in self.game_metadata:
                        yield game_id, game_json
                    else:
                        logging.debug(f"Skipping game id not in metadata: {game_id}")

        encoded_game_tuples = joblib.Parallel(n_jobs=cfg.num_dataloader_workers)(
            joblib.delayed(encode_game)(
                game_id,
                game_json,
                value_dir=self.value_dir,
                only_with_min_final_score=self.only_with_min_final_score,
                cf_agent=self.cf_agent,
                n_cf_agent_samples=self.n_cf_agent_samples,
                input_valid_power_idxs=self.get_valid_power_idxs(game_id),
                game_metadata=None if self.game_metadata is None else self.game_metadata[game_id],
                exclude_n_holds=self.exclude_n_holds,
                all_powers=self.all_powers,
                input_version=self.input_version,
                return_hold_for_invalid=self.return_hold_for_invalid,
                max_year=self.max_year,
            )
            for game_id, game_json in tqdm(read_data(paths_and_jsons), total=n_games)
        )

        # Filter for games with valid phases
        encoded_games = [
            g for g in encoded_game_tuples if g is not None and g["valid_power_idxs"][0].any()
        ]
        logging.info(f"Found valid data for {len(encoded_games)} / {n_games} games")

        # Build x_idx and power_idx tensors used for indexing
        power_idxs, x_idxs = [], []
        x_idx = 0
        for encoded_game in encoded_games:
            for valid_power_idxs in encoded_game["valid_power_idxs"]:
                assert valid_power_idxs.nelement() == len(POWERS), (
                    encoded_game["valid_power_idxs"].shape,
                    valid_power_idxs.shape,
                )
                for power_idx in valid_power_idxs.nonzero(as_tuple=False)[:, 0]:
                    power_idxs.append(power_idx)
                    x_idxs.append(x_idx)
                x_idx += 1

        self.power_idxs = torch.tensor(power_idxs, dtype=torch.long)
        self.x_idxs = torch.tensor(x_idxs, dtype=torch.long)

        # now collate the data into giant tensors!
        self.encoded_games = DataFields.cat(encoded_games)

        self.num_games = len(encoded_games)
        self.num_phases = len(self.encoded_games["x_board_state"]) if self.encoded_games else 0
        self.num_elements = len(self.x_idxs)

        self.validate_dataset()
        logging.info("Validated dataset, returning")

    def stats_str(self):
        return f"Dataset: {self.num_games} games, {self.num_phases} phases, and {self.num_elements} elements."

    def validate_dataset(self):
        assert len(self) > 0
        for e in self.encoded_games.values():
            if isinstance(e, TensorList):
                max_seq_len = N_SCS if self.all_powers else MAX_SEQ_LEN
                assert len(e) == self.num_phases * len(POWERS) * max_seq_len
            else:
                assert len(e) == self.num_phases

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.long)

        max_seq_len = N_SCS if self.all_powers else MAX_SEQ_LEN

        assert isinstance(idx, torch.Tensor) and idx.dtype == torch.long
        assert idx.max() < len(self)

        sample_idxs = idx % self.n_cf_agent_samples
        idx //= self.n_cf_agent_samples
        assert isinstance(idx, torch.Tensor)

        x_idx = self.x_idxs[idx]
        power_idx = self.power_idxs[idx]

        fields = self.encoded_games.select(x_idx)  # [x[x_idx] for x in self.encoded_games[:-1]]

        if self.all_powers:
            # non-A phases are encoded in power idx 0
            power_idx[~fields["x_in_adj_phase"].bool()] = 0

        # unpack the possible_actions
        possible_actions_idx = ((x_idx * len(POWERS) + power_idx) * max_seq_len).unsqueeze(
            1
        ) + torch.arange(max_seq_len).unsqueeze(0)
        x_possible_actions = self.encoded_games["x_possible_actions"][
            possible_actions_idx.view(-1)
        ]
        x_possible_actions_padded = x_possible_actions.to_padded(
            total_length=MAX_VALID_LEN, padding_value=EOS_IDX
        )
        fields["x_possible_actions"] = x_possible_actions_padded.view(
            len(idx), max_seq_len, MAX_VALID_LEN
        ).long()

        # for these fields we need to select out the correct power
        for f in ("x_power", "x_loc_idxs", "y_actions"):
            fields[f] = fields[f][torch.arange(len(fields[f])), power_idx]

        # for y_actions, select out the correct sample
        fields["y_actions"] = (
            fields["y_actions"]
            .gather(1, sample_idxs.view(-1, 1, 1).repeat((1, 1, fields["y_actions"].shape[2])))
            .squeeze(1)
        )

        fields["x_loc_idxs"] = fields["x_loc_idxs"].float()

        # set all_powers
        fields["all_powers"] = self.all_powers

        return fields

    def __len__(self):
        return self.num_elements * self.n_cf_agent_samples

    def get_valid_power_idxs(self, game_id) -> List[bool]:
        if self.game_metadata is None:
            return [True for pwr in POWERS]
        game_meta = self.game_metadata[game_id]
        return [
            (
                game_meta[pwr]["rating_percentile"] >= self.min_rating_percentile
                and game_meta[pwr]["total"] >= self.min_total_games
            )
            for pwr in POWERS
        ]


def get_paths_and_jsons(
    game_data_path: Union[str, pathlib.Path], limit_n_games: Optional[int]
) -> List[Tuple[str, str]]:
    """
    If game_data_path is a file, expects it to be lines of <path> <json>.

    If game_data_path is a dir, expects that dir to contain a bunch of game jsons.

    Either way, returns list of tuples (path, json)
    """
    paths_and_jsons = []
    if os.path.isdir(game_data_path):
        files = os.listdir(game_data_path)
        files = [file for file in files if file.endswith(".json")]
        files = sorted(files)
        for i, file in enumerate(files):
            if limit_n_games is not None and limit_n_games >= 0 and i >= limit_n_games:
                break
            game_path = os.path.join(game_data_path, file)
            with open(game_path) as f:
                game_json = f.read()
            paths_and_jsons.append((game_path, game_json))
    else:
        with open(game_data_path) as f:
            for i, line in enumerate(f.readlines()):
                if limit_n_games is not None and limit_n_games >= 0 and i >= limit_n_games:
                    break
                game_path, game_json = line.split(" ", 1)
                paths_and_jsons.append((game_path, game_json))
    return paths_and_jsons


def encode_game(
    game_id,
    game_json: str,
    *,
    only_with_min_final_score=7,
    value_dir: Optional[str],
    cf_agent=None,
    n_cf_agent_samples=1,
    input_valid_power_idxs,
    game_metadata: Optional[Dict[str, Any]],
    exclude_n_holds,
    all_powers: bool,
    input_version: int = DEFAULT_INPUT_VERSION,
    return_hold_for_invalid: bool,
    max_year: Optional[int],
) -> Optional[DataFields]:
    torch.set_num_threads(1)
    encoder = FeatureEncoder()
    try:
        game = Game.from_json(game_json)
    except RuntimeError:
        logging.debug(f"RuntimeError (json decoding) while loading game id {game_id}")
        return None

    phase_names = [phase.name for phase in game.get_phase_history()]
    loaded_power_values = load_power_values(value_dir, game_id)

    num_phases = len(phase_names)
    logging.info(f"Encoding {game.game_id} with {num_phases} phases")
    phase_encodings = []

    for phase_idx in range(num_phases):
        if max_year is not None and max_year >= 1900:
            try:
                year = year_of_phase(phase_names[phase_idx])
            except ValueError:
                year = None
            if year is None or year > max_year:
                continue

        phase_encoding = encode_phase(
            encoder,
            game,
            game_id,
            phase_idx,
            phase_names[phase_idx],
            only_with_min_final_score=only_with_min_final_score,
            cf_agent=cf_agent,
            n_cf_agent_samples=n_cf_agent_samples,
            input_valid_power_idxs=input_valid_power_idxs,
            exclude_n_holds=exclude_n_holds,
            loaded_power_values=loaded_power_values,
            all_powers=all_powers,
            input_version=input_version,
            return_hold_for_invalid=return_hold_for_invalid,
        )
        phase_encodings.append(phase_encoding)

    num_phases = len(phase_encodings)
    stacked_encodings = DataFields.cat(phase_encodings)

    has_press = torch.zeros(num_phases, 1)
    if game_metadata is not None:
        has_press = has_press + (1 if game_metadata["press_type"] != "NoPress" else 0)
    stacked_encodings["x_has_press"] = has_press

    player_ratings = torch.zeros(num_phases, len(POWERS))
    if game_metadata is not None:
        for i, pwr in enumerate(POWERS):
            player_ratings[:, i] = game_metadata[pwr]["rating_percentile"]
    stacked_encodings["x_player_ratings"] = player_ratings

    return stacked_encodings


@dataclass
class PrecomputedGameValues:
    # The target value that you should train to, such as an exponential weighted average
    # of model values over future states
    sos_values_by_phase: Dict[Phase, Dict[Power, float]]
    dss_values_by_phase: Dict[Phase, Dict[Power, float]]
    # The current raw value at each phase according to some model
    sos_raw_model_values_by_phase: Optional[Dict[Phase, Dict[Power, float]]]
    dss_raw_model_values_by_phase: Optional[Dict[Phase, Dict[Power, float]]]

    @classmethod
    def of_json_data(cls, state: Dict[str, Any]) -> "PrecomputedGameValues":
        # Legacy format is that the json is just the raw dict from str to float
        if "version" not in state or state["version"] == 1:
            return PrecomputedGameValues(
                sos_values_by_phase=state,
                dss_values_by_phase=state,
                sos_raw_model_values_by_phase=None,
                dss_raw_model_values_by_phase=None,
            )
        assert state["version"] == 2
        return PrecomputedGameValues(
            sos_values_by_phase=state["sos_values_by_phase"],
            dss_values_by_phase=state["dss_values_by_phase"],
            sos_raw_model_values_by_phase=state["sos_raw_model_values_by_phase"],
            dss_raw_model_values_by_phase=state["dss_raw_model_values_by_phase"],
        )

    def to_json_data(self) -> Dict[str, Any]:
        state = {}
        state["version"] = 2
        state["sos_values_by_phase"] = self.sos_values_by_phase
        state["dss_values_by_phase"] = self.dss_values_by_phase
        state["sos_raw_model_values_by_phase"] = self.sos_raw_model_values_by_phase
        state["dss_raw_model_values_by_phase"] = self.dss_raw_model_values_by_phase
        return state


def load_power_values(value_dir: Optional[str], game_id: int) -> Optional[PrecomputedGameValues]:
    """If value_dir is not None, tries to load values from a json values file.

    Returns:
        - A map from phase name to {pwr: value}
    """
    if value_dir is not None:
        value_path = os.path.join(value_dir, f"game_{game_id}.json")
        try:
            with open(value_path) as f:
                return PrecomputedGameValues.of_json_data(json.load(f))
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f"Error while loading values at {value_path}: {e}")
    return None


def encode_phase(
    encoder: FeatureEncoder,
    game: Game,
    game_id: str,
    phase_idx: int,
    phase_name: str,
    *,
    only_with_min_final_score: Optional[int],
    cf_agent=None,
    n_cf_agent_samples=1,
    input_valid_power_idxs,
    exclude_n_holds,
    loaded_power_values: Optional[PrecomputedGameValues] = None,
    all_powers: bool,
    input_version: int,
    return_hold_for_invalid: bool,
):
    """
    Arguments:
    - game: Game object
    - game_id: unique id for game
    - phase_idx: int, the index of the phase to encode
    - only_with_min_final_score: if specified, only encode for powers who
      finish the game with some # of supply centers (i.e. only learn from
      winners). MILA uses 7.

    Returns: DataFields, including y_actions and (y_final_score or sos_scores and dss_scores)
    """

    # keep track of which powers are invalid, and which powers are valid but
    # weak (by min-rating and min-score)
    strong_power_idxs = torch.tensor(input_valid_power_idxs, dtype=torch.bool)
    valid_power_idxs = torch.ones_like(strong_power_idxs, dtype=torch.bool)

    phase = game.get_phase_history()[phase_idx]

    rolled_back_game = game.rolled_back_to_phase_start(phase.name)

    encode_fn = encoder.encode_inputs_all_powers if all_powers else encoder.encode_inputs
    data_fields = encode_fn([rolled_back_game], input_version=input_version)

    # encode final scores
    if loaded_power_values is not None:
        data_fields["sos_scores"] = torch.tensor(
            [loaded_power_values.sos_values_by_phase[phase_name][power] for power in POWERS]
        ).unsqueeze(0)
        data_fields["dss_scores"] = torch.tensor(
            [loaded_power_values.dss_values_by_phase[phase_name][power] for power in POWERS]
        ).unsqueeze(0)
    else:
        data_fields["sos_scores"] = torch.tensor([game.get_scores(Game.SCORING_SOS)])
        data_fields["dss_scores"] = torch.tensor([game.get_scores(Game.SCORING_DSS)])

    EMPTY_ACTION: Action = tuple()  # force type of tuple to action to make typechecking happy

    # get actions from phase, or from cf_agent if set
    joint_action_samples = (
        {power: [phase.orders.get(power, EMPTY_ACTION)] for power in POWERS}
        if cf_agent is None
        else get_cf_agent_order_samples(rolled_back_game, phase.name, cf_agent, n_cf_agent_samples)
    )

    # encode y_actions
    max_seq_len = N_SCS if all_powers else MAX_SEQ_LEN
    y_actions = torch.full(
        (len(POWERS), n_cf_agent_samples, max_seq_len), EOS_IDX, dtype=torch.long
    )
    joint_action: Dict[Power, Action] = {}
    for sample_i in range(n_cf_agent_samples):
        joint_action = {
            power: action_samples[sample_i]
            for power, action_samples in joint_action_samples.items()
        }
        if data_fields["x_in_adj_phase"] or not all_powers:
            for power_i, power in enumerate(POWERS):
                y_actions[power_i, sample_i, :], valid = encode_power_actions(
                    joint_action.get(power, EMPTY_ACTION),
                    data_fields["x_possible_actions"][0, power_i],
                    data_fields["x_in_adj_phase"][0],
                    max_seq_len=max_seq_len,
                    return_hold_for_invalid=return_hold_for_invalid,
                )
                valid_power_idxs[power_i] &= valid
        else:
            # do all_powers encoding
            y_actions[0, sample_i], valid_mask = encode_all_powers_action(
                joint_action,
                data_fields["x_possible_actions"][0],
                data_fields["x_power"][0],
                data_fields["x_in_adj_phase"][0],
                return_hold_for_invalid=return_hold_for_invalid,
            )
            valid_power_idxs &= valid_mask

    # Check for all-holds, no orders
    for power_i, power in enumerate(POWERS):
        orders = joint_action.get(power, EMPTY_ACTION)
        if len(orders) == 0 or (
            0 <= exclude_n_holds <= len(orders) and all(order.endswith(" H") for order in orders)
        ):
            valid_power_idxs[power_i] = 0

    # Maybe filter away powers that don't finish with enough SC.
    # If all players finish with fewer SC, include everybody.
    # cf. get_top_victors() in mila's state_space.py
    if only_with_min_final_score is not None:
        final_score = {k: len(v) for k, v in game.get_state()["centers"].items()}
        if max(final_score.values()) >= only_with_min_final_score:
            for i, power in enumerate(POWERS):
                if final_score.get(power, 0) < only_with_min_final_score:
                    strong_power_idxs[i] = 0

    data_fields["y_actions"] = y_actions.unsqueeze(0)
    data_fields["x_possible_actions"] = TensorList.from_padded(
        data_fields["x_possible_actions"].view(-1, MAX_VALID_LEN), padding_value=EOS_IDX
    )
    data_fields["valid_power_idxs"] = valid_power_idxs.unsqueeze(0) & strong_power_idxs
    data_fields["valid_power_idxs_any_strength"] = valid_power_idxs.unsqueeze(0)

    if not all_powers:
        data_fields["x_power"] = torch.arange(len(POWERS)).view(1, -1, 1).repeat(1, 1, MAX_SEQ_LEN)

    return data_fields


def encode_power_actions(
    orders: Action,
    x_possible_actions,
    x_in_adj_phase,
    *,
    max_seq_len=MAX_SEQ_LEN,
    return_hold_for_invalid=False,
) -> Tuple[torch.Tensor, bool]:
    """
    Arguments:
    - a tuple of orders, e.g. ("F APU - ION", "A NAP H")
    - x_possible_actions, a LongTensor of valid actions for this power-phase, shape=[17, 469]
    Returns a tuple:
    - max_seq_len-len 1d-tensor, pad=EOS_IDX
    - True/False is valid
    """
    y_actions = torch.full((max_seq_len,), EOS_IDX, dtype=torch.int32)
    order_idxs = []

    # Check for missing unit orders
    if not x_in_adj_phase:
        n_expected = len([x for x in x_possible_actions[:, 0].tolist() if x != -1])
        if n_expected != len(orders):
            logging.debug(f"Missing orders: {orders}, n_expected={n_expected}")
            return y_actions, False

    if any(len(order.split()) < 3 for order in orders):
        # skip over power with unparseably short order
        return y_actions, False
    elif any(order.split()[2] == "B" for order in orders):
        try:
            order_idxs.extend(action_strs_to_global_idxs(orders))
        except Exception:
            logging.debug(f"Invalid build orders: {orders}")
            return y_actions, False
    else:
        try:
            order_idxs.extend(
                action_strs_to_global_idxs(
                    orders,
                    try_strip_coasts=True,
                    ignore_missing=False,
                    return_hold_for_invalid=return_hold_for_invalid,
                    sort_by_loc=True,
                )
            )
        except OrderIdxConversionException:
            logging.debug(f"Bad order in: {orders}")
            return y_actions, False

    for i, order_idx in enumerate(order_idxs):
        try:
            cand_idx = (x_possible_actions[i] == order_idx).nonzero(as_tuple=False)[0, 0]
            y_actions[i] = cand_idx
        except IndexError:
            # filter away powers whose orders are not in valid_orders
            # most common reasons why this happens:
            # - actual garbage orders (e.g. moves between non-adjacent locations)
            # - too many orders (e.g. three build orders with only two allowed builds)
            return y_actions, False

    return y_actions, True


def encode_all_powers_action(
    joint_action: JointAction,
    x_possible_actions: torch.Tensor,
    x_power: torch.Tensor,
    x_in_adj_phase,
    *,
    return_hold_for_invalid=False,
):
    """
    Encode y_actions with all_powers=True. Some powers may be invalid due e.g.
    to missing orders, but we take care to ensure that y_actions lines up with
    x_possible_actions despite possible missing or invalid orders.

    Returns a tuple:
    - y_actions, a 1-d sequence of local order idxs in global LOC order
    - valid_mask: a [7]-shaped bool tensor indicating whether each power's actions were valid
    """
    max_seq_len = N_SCS
    valid_mask = torch.ones(7, dtype=torch.bool)
    y_actions = torch.full((max_seq_len,), EOS_IDX, dtype=torch.long)

    assert not x_in_adj_phase, "Use encode_power_actions"
    assert x_possible_actions.shape[0] == len(POWERS), x_possible_actions.shape
    assert x_possible_actions.shape[1] == max_seq_len, x_possible_actions.shape

    assert (x_possible_actions[1:] == EOS_IDX).all()
    x_possible_actions = x_possible_actions[0]

    orders_by_loc = {
        order.split()[1]: order for power, orders in joint_action.items() for order in orders
    }
    expected_locs_sorted = [
        order.split()[1] for order in global_order_idxs_to_str(x_possible_actions[:, 0])
    ]
    orders_sorted = [orders_by_loc.get(loc) for loc in expected_locs_sorted]
    # We don't pass sort_by_loc to action_strs_to_global_idxs even though that's what we would normally
    # do to be consistent with dipcc's encoding. Instead, we manually sort ourselves via orders_sorted,
    # because we've inserted None in some spots (when orders_by_loc.get(loc) doesn't find the loc)
    # and so action_strs_to_global_idxs wouldn't have enough information to know the correct ordering.
    global_idxs = action_strs_to_global_idxs(
        orders_sorted,
        try_strip_coasts=True,
        return_none_for_missing=True,
        return_hold_for_invalid=return_hold_for_invalid,
    )

    for step, (order, loc, global_idx) in enumerate(
        zip(orders_sorted, expected_locs_sorted, global_idxs)
    ):
        if order is None:
            # Missing order
            valid_mask[x_power[0, step]] = 0
            continue

        if global_idx is None:
            # Handled OrderIdxConversionException
            valid_mask[x_power[0, step]] = 0
            continue

        nz = (x_possible_actions[step] == global_idx).nonzero(as_tuple=True)
        if len(nz) != 1:
            got_loc = order.split()[1][:3]
            assert got_loc == loc, f"{got_loc} != {loc}"
            logging.debug(f"Unexpected order for {loc}: {order}")
            valid_mask[x_power[0, step]] = 0
            continue
        local_idx = nz[0][0].item()
        y_actions[step] = local_idx

    return y_actions, valid_mask


# imported by RL code
def cat_pad_inputs(xs: List[DataFields]) -> DataFields:
    batch = DataFields({k: [x[k] for x in xs] for k in xs[0].keys()})
    for k, v in batch.items():
        if k == "x_possible_actions":
            batch[k] = cat_pad_sequences(v, pad_value=-1, pad_to_len=MAX_SEQ_LEN)
        elif k == "x_loc_idxs":
            batch[k] = cat_pad_sequences(v, pad_value=EOS_IDX, pad_to_len=MAX_SEQ_LEN)
        else:
            batch[k] = torch.cat(v)

    return batch


def shuffle_locations(batch: DataFields) -> DataFields:
    """Change location order in the batch randomly."""
    x_loc_idxs = batch["x_loc_idxs"]
    *batch_dims, _ = x_loc_idxs.shape
    device = x_loc_idxs.device

    loc_priority = torch.rand((*batch_dims, MAX_SEQ_LEN), device=device)
    # Shape: [batch_dims, 1]. Note, this will return 0 in case of adjustment
    # phase. That's a safe bet as we don't know how many actions will actually
    # present in y_actions.
    num_locs = (x_loc_idxs >= 0).sum(-1, keepdim=True)
    unsqueeze_shape = [1] * len(batch_dims) + [-1]
    invalid_mask = (
        torch.arange(MAX_SEQ_LEN, device=device).view(unsqueeze_shape).expand_as(loc_priority)
        >= num_locs
    )
    loc_priority += (1000 + torch.arange(MAX_SEQ_LEN, device=device)) * invalid_mask.float()
    perm = loc_priority.sort(dim=-1).indices
    return reorder_locations(batch, perm)


def reorder_locations(batch: DataFields, perm: torch.Tensor) -> DataFields:
    """Change location order in the batch according to the permutation."""
    batch = batch.copy()
    if "y_actions" in batch:
        y_actions = batch["y_actions"]
        batch["y_actions"] = y_actions.gather(-1, perm[..., : y_actions.shape[-1]])
        assert (
            (y_actions == -1) == (batch["y_actions"] == -1)
        ).all(), "permutation must keep undefined locations in place"

    if len(batch["x_possible_actions"].shape) == 3:
        batch["x_possible_actions"] = batch["x_possible_actions"].gather(
            -2, perm.unsqueeze(-1).repeat(1, 1, 469)
        )
    else:
        assert len(batch["x_possible_actions"].shape) == 4
        batch["x_possible_actions"] = batch["x_possible_actions"].gather(
            -2, perm.unsqueeze(-1).repeat(1, 1, 1, 469)
        )

    # In case if a move phase, x_loc_idxs is B x 81 or B x 7 x 81, where the
    # value in each loc is which order in the sequence it is (or -1 if not in
    # the sequence).
    new_x_loc_idxs = batch["x_loc_idxs"].clone()
    for lidx in range(perm.shape[-1]):
        mask = batch["x_loc_idxs"] == perm[..., lidx].unsqueeze(-1)
        new_x_loc_idxs[mask] = lidx
    batch["x_loc_idxs"] = new_x_loc_idxs
    return batch


def maybe_augment_targets_inplace(
    batch: DataFields,
    *,
    single_chances: Optional[float],
    double_chances: Optional[float],
    six_chances: Optional[float] = None,
    power_conditioning: Optional[conf_cfgs.PowerConditioning],
    debug_print: bool = False,
    input_version: int = DEFAULT_INPUT_VERSION,
) -> None:
    """Augment allpower batch: sub-samples powers and adds contitioning.

    Power subsampling:
     select only data related to one or two powers with some probability.

    Let all_chances = 1 and Z = all_chances + single_chances + double_chances + six_chances.

    Then with probability:
      all_chances / Z a training example will be kept intact
      single_chances / Z a single power will be sampled from a training example
      double_chances / Z a pair of powers will be sampled from a training example
      six_chances / Z a sixtuple of powers will be sampled from a training example

    The sampling applied to each element within the batch independently.
    If the training example is for adjustment phase or doesn't contain enough
    powers we will keep it intact.


    Conditioning:
        With probability single_power_conditioning_prob will add this_orders
        tensor with orders of some single power to condition on.

        Conditioning only applies for all-power batches

    """
    if debug_print:
        orig_batch = copy.deepcopy(batch)
    else:
        orig_batch = None
    batch_size = len(batch["y_actions"])
    x_loc_idxs_orig = batch["x_loc_idxs"].cpu().clone()
    # How many powers to subsample. 0 means do not subsample.
    probs = np.array([1.0, single_chances or 0.0, double_chances or 0.0, six_chances or 0.0])
    batch_num_subsampled_powers = np.random.choice(
        np.array([0, 1, 2, 6]), p=probs / probs.sum(), size=[batch_size]
    )

    if len(batch["x_power"].shape) == 2:
        # TrainSL "flattens" input so that we don't get the power dimension.
        assert batch["x_power"].shape == (batch_size, N_SCS), batch["x_power"].shape
        assert batch["x_loc_idxs"].shape == (batch_size, len(LOCS)), batch["x_loc_idxs"].shape
        assert batch["y_actions"].shape == (batch_size, N_SCS), batch["y_actions"].shape
        flattened_input = True
    else:
        # RL gives data as FeatyreEncoder returns it.
        assert batch["x_power"].shape == (batch_size, len(POWERS), N_SCS), batch["x_power"].shape
        assert batch["x_loc_idxs"].shape == (batch_size, len(POWERS), len(LOCS)), batch[
            "x_loc_idxs"
        ].shape
        assert batch["y_actions"].shape == (batch_size, len(POWERS), N_SCS), batch[
            "y_actions"
        ].shape
        flattened_input = False

    # ---- Power subsampling.
    for i, num_subsampled_powers in enumerate(batch_num_subsampled_powers):
        if num_subsampled_powers == 0:
            continue
        assert num_subsampled_powers in (1, 2, 6)
        if not (x_loc_idxs_orig[i] > 0).any():
            # Either builds or not orders. Either way it has no more than 1 power.
            continue
        alive_powers = frozenset(
            x
            # This works for both flattened and non-flattened input.
            for x in batch["x_power"][i][batch["y_actions"][i] != EOS_IDX].tolist()
            if x != EOS_IDX
        )
        if len(alive_powers) <= num_subsampled_powers:
            # We already have no more than num_subsampled_powers powers, so no need to do anything.
            continue
        selected_powers = batch["x_power"].new_tensor(
            np.random.choice(sorted(alive_powers), size=[num_subsampled_powers], replace=False)
        )
        if flattened_input:
            select_powers_inplace(
                selected_powers,
                batch["x_power"][i],
                batch["x_possible_actions"][i],
                batch["x_loc_idxs"][i],
                y_actions=batch["y_actions"][i],
            )
        else:
            power_id = 0  # Allpower so all other powers are EOS_IDX.
            select_powers_inplace(
                selected_powers,
                batch["x_power"][i][power_id],
                batch["x_possible_actions"][i][power_id],
                batch["x_loc_idxs"][i][power_id],
                y_actions=batch["y_actions"][i][power_id],
            )

        if debug_print:
            assert orig_batch is not None
            print("@" * 80)
            print("@" * 80)
            print("num_subsampled_powers =", num_subsampled_powers)
            explain_base_strategy_model_decoder_inputs(
                loc_idxs=orig_batch["x_loc_idxs"][i : i + 1],
                all_cand_idxs=orig_batch["x_possible_actions"][i : i + 1],
                power=orig_batch["x_power"][i : i + 1],
                teacher_force_orders=orig_batch["y_actions"][i : i + 1],
                teacher_forces_global=False,
            )
            explain_base_strategy_model_decoder_inputs(
                loc_idxs=batch["x_loc_idxs"][i : i + 1],
                all_cand_idxs=batch["x_possible_actions"][i : i + 1],
                power=batch["x_power"][i : i + 1],
                teacher_force_orders=batch["y_actions"][i : i + 1],
                teacher_forces_global=False,
            )

    # ---- Conditioning.
    if power_conditioning is not None:
        assert power_conditioning.min_num_power is not None
        assert power_conditioning.max_num_power is not None
        assert power_conditioning.prob is not None

        batch_this_orders = batch["x_prev_orders"].new_zeros(batch["x_prev_orders"].shape)
        feature_encoder = FeatureEncoder()
        y_actions_global = local_order_idxs_to_global(
            batch["y_actions"], batch["x_possible_actions"],
        )
        for i, num_subsampled_powers in enumerate(batch_num_subsampled_powers):
            if num_subsampled_powers != 0:
                # Only working with all-power batch elements.
                continue
            if not (x_loc_idxs_orig[i] > 0).any():
                # Either builds or not orders. Either way it has no more than 1 power.
                continue
            if np.random.random() > power_conditioning.prob:
                continue
            alive_powers = frozenset(
                x
                for x in batch["x_power"][i][batch["y_actions"][i] != EOS_IDX].tolist()
                if x != EOS_IDX
            )
            num_condition_power = np.random.randint(
                power_conditioning.min_num_power,
                min(power_conditioning.max_num_power, len(alive_powers)) + 1,
            )
            condition_powers = np.random.choice(
                sorted(alive_powers), size=num_condition_power, replace=False
            )

            # Global order ids.
            condition_action_ids = []
            for power in condition_powers:
                # This works for both flattened and non-flattened input.
                condition_action_ids.append(y_actions_global[i][batch["x_power"][i] == power])
            condition_action_ids = torch.cat(condition_action_ids, 0)
            condition_action_strs = global_order_idxs_to_str(condition_action_ids)

            # Strict encoding should be fine here because the condition_action_strs were
            # themselves generated from global_order_idxs
            batch_this_orders[i] = feature_encoder.encode_orders_single_strict(
                condition_action_strs, input_version
            )
        batch["x_current_orders"] = batch_this_orders


def select_powers_inplace(
    selected_powers: torch.Tensor,
    x_power: torch.Tensor,
    x_possible_actions: torch.Tensor,
    x_loc_idxs: torch.Tensor,
    *,
    y_actions: Optional[torch.Tensor] = None,
):
    """Modifies allpower inputs/outputs to keep only values related to the powers."""
    assert x_power.shape == (N_SCS,), x_power.shape
    selected_timestamps_mask = (x_power[:, None] == selected_powers[None, :]).any(-1)
    num_locs = selected_timestamps_mask.sum()

    if y_actions is not None:
        y_actions[:num_locs] = y_actions[selected_timestamps_mask]
        y_actions[num_locs:] = EOS_IDX

    x_possible_actions[:num_locs] = x_possible_actions[selected_timestamps_mask]
    x_possible_actions[num_locs:] = EOS_IDX

    # Set all location to selected_power. To need to mask, it will be masked by y_actions.
    x_power[:num_locs] = x_power[selected_timestamps_mask]
    x_power[num_locs:] = EOS_IDX

    # E.g. [False, True, False, False, True, True] -> [ -1, 0, -1, -1, 1, 2 ]
    assert EOS_IDX == -1
    assert len(selected_timestamps_mask.shape) == 1
    assert len(x_loc_idxs.shape) == 1
    assert x_power.shape[0] == selected_timestamps_mask.shape[0]
    new_timestamp_of_old_timestamp = (
        torch.cumsum(selected_timestamps_mask.long(), dim=0) * selected_timestamps_mask - 1
    )
    # E.g. [ -1, 0, -1, -1, 1, 2 ] -> [ -1, -1, 0, -1, -1, 1, 2 ]
    # To handle EOS_IDX
    new_timestamp_of_old_timestamp_padded = torch.nn.functional.pad(
        new_timestamp_of_old_timestamp, (1, 0), mode="constant", value=EOS_IDX
    )
    # E.g.
    # x_loc_idxs = [ -1, 2, 4, -1 ]
    # new_timestamp_of_old_timestamp = [ 7, -1, 6, 3, 5 ]
    # new_timestamp_of_old_timestamp_padded = [ -1, 7, -1, 6, 3, 5 ]
    # x_loc_idxs (afterward) = [ -1, 6, 5, -1 ]
    x_loc_idxs[:] = torch.gather(
        new_timestamp_of_old_timestamp_padded, dim=0, index=(x_loc_idxs + 1).long()
    )


def get_cf_agent_order_samples(game, phase_name, cf_agent, n_cf_agent_samples):
    assert game.get_state()["name"] == phase_name, f"{game.get_state()['name']} != {phase_name}"

    if hasattr(cf_agent, "run_search"):
        # agent_state=None should fail on any agent that needs state as part of its search
        # ... which is fine - if we ever need to cf annotate a dataset with such an agent
        # we can figure it out then.
        policies = cf_agent.run_search(game, agent_state=None).get_agent_policy()
        logging.info(f"run_search: {policies}")
        return {
            power: (
                [sample_p_dict(policies[power]) for _ in range(n_cf_agent_samples)]
                if policies[power]
                else []
            )
            for power in POWERS
        }
    else:
        # Agents that require nontrivial state currently not supported
        state = NoAgentState()
        return {
            power: [cf_agent.get_orders(game, power, state) for _ in range(n_cf_agent_samples)]
            for power in POWERS
        }


def add_rating_percentiles_to_metadata(metadata: Dict):
    ratings = torch.tensor(
        [
            game[pwr]["logit_rating"]
            for game in metadata.values()
            for pwr in POWERS
            if (pwr in game and game[pwr] is not None)
        ]
    )
    sorted_ratings, rating_order = ratings.sort()
    # rating order says where each sorted rating *comes from*
    # but to get the percentile we need to know where each rating
    # *goes* in the order
    N = len(rating_order)
    rating_percentile = torch.empty(N)
    rating_percentile[rating_order] = torch.arange(N, dtype=torch.float) / N

    idx = 0
    for game in metadata.values():
        for pwr in POWERS:
            if pwr in game and game[pwr] is not None:
                assert (
                    abs(game[pwr]["logit_rating"] - ratings[idx]) < 1e-3
                ), f"{game[pwr]['logit_rating']} {ratings[idx]}"
                game[pwr]["rating_percentile"] = float(rating_percentile[idx])
                idx += 1
    assert idx == len(rating_percentile)


def is_int(k):
    try:
        int(k)
        return True
    except ValueError:
        return False
