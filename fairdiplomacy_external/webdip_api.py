#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
from collections import defaultdict
from fnmatch import fnmatch
import gc
import html
import http.client
import math
import pickle
import traceback
from requests.models import Response
import requests
import socket
import urllib3.exceptions
from fairdiplomacy.agents.parlai_message_handler import (
    ParlaiMessageHandler,
    pseudoorders_initiate_sleep_heuristics_should_trigger,
)
from fairdiplomacy.agents.player import Player
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.typedefs import (
    Json,
    MessageDict,
    MessageHeuristicResult,
    OutboundMessageDict,
    Phase,
    Power,
    Timestamp,
    Context,
)
import random
import hashlib
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import getpass
import itertools
import json
import logging
import os
import pathlib
import time
from fairdiplomacy.data.build_dataset import (
    DRAW_VOTE_TOKEN,
    UNDRAW_VOTE_TOKEN,
    DATASET_DRAW_MESSAGE,
    DATASET_NODRAW_MESSAGE,
)
from fairdiplomacy.utils.agent_interruption import ShouldStopException, set_interruption_condition
from fairdiplomacy.utils.atomicish_file import atomicish_open_for_writing_binary
from fairdiplomacy.utils.slack import GLOBAL_SLACK_EXCEPTION_SWALLOWER
from fairdiplomacy.utils.typedefs import build_message_dict, get_last_message
from fairdiplomacy.viz.meta_annotations.annotator import MetaAnnotator
from parlai_diplomacy.utils.game2seq.format_helpers.misc import POT_TYPE_CONVERSION
import torch
from fairdiplomacy.utils.game import game_from_view_of
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.pydipcc import Game
from fairdiplomacy.webdip.message_approval_cache_api import (
    PRESS_COUNTRY_ID_TO_POWER,
    ApprovalStatus,
    MessageReviewData,
    compute_phase_message_history_state_with_power,
    flag_proposal_as_stale,
    gen_id,
    get_message_review,
    get_redis_host,
    get_should_run_backup,
    get_kill_switch,
    maybe_get_phase_message_history_state,
    set_message_review,
    delete_message_review,
    botgame_fp_to_context,
    update_phase_message_history_state,
)
from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.data.build_dataset import (
    GameVariant,
    TERR_ID_TO_LOC_BY_MAP,
    COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP,
    COUNTRY_POWER_TO_ID,
    get_valid_coastal_variant,
)
from fairdiplomacy.webdip.utils import turn_to_phase
from fairdiplomacy.utils.slack import send_slack_message
import heyhi
from conf import conf_cfgs
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME

from fairdiplomacy.agents.searchbot_agent import SearchBotAgentState

######

# This import triggers a GPL license for this file.
# As a result, we release the fairdiplomacy_external subdirectory under a GPL license,
# while releasing the rest of the repository under an MIT license.
from diplomacy.integration.webdiplomacy_net.orders import Order as WebdipOrder

######


"""
Some hard-coded global constants need to be set for this work. See fairdiplomacy/webdip/message_approval_cache_api.py.
"""

GameId = int
DialogueState = Tuple[str, int]  # phase, num_messages
API_PATH = "/api.php"
WEBDIP_URL = "https://webdiplomacy.net/"

GLOBAL_CHAT_COUNTRY_ID = 0
STATUS_ROUTE = "game/status"
MISSING_ORDERS_ROUTE = "players/missing_orders"
ACTIVE_GAMES_ROUTE = "players/active_games"
POST_ORDERS_ROUTE = "game/orders"
SEND_MESSAGE_ROUTE = "game/sendmessage"
VOTE_ROUTE = "game/togglevote"

logger = logging.getLogger("webdip")

GAME_NOT_FOUND_RESP = b"<html><head><title>webDiplomacy fatal error</title></head>\r\n\t\t\t\t<body><p>Error occurred during script startup, usually a result of inability to connect to the database:</p>\r\n\t\t\t\t<p>Game not found, or has an invalid variant set; ensure a valid game ID has been given. Check that this game hasn't been canceled, you may have received a message about it on your <a href='index.php' class='light'>home page</a>.</p></body></html>"
ACCESS_DENIED_RESP = b"Access to this page denied for your account type."

RETRY_SLEEP_TIME = 10
RETRY_SUCCESS_TIME = 120  # If no error for this many second, then we assume everything is good
MAX_RETRIES_FOR_NETWORK_ISSUES = 5

RECONNECT_SLEEP_TIMES = [5, 10, 30, 60, 120, 300]

MESSAGE_DELAY_IF_SLEEP_INF = Timestamp.from_seconds(60)


class UnexpectedWebdipBehaviorException(Exception):
    pass


class WebdipSendMessageFailedException(UnexpectedWebdipBehaviorException):
    pass


RA_PHASE_MESSAGE_SEND_IN_RULEBOOK_ERROR_MESSAGE = (
    "Message send failed: b'Message is invalid in RulebookPress'"
)


class WebdipGameNotFoundException(Exception):
    pass


def phase_to_turn(phase: Phase) -> int:
    year = int(phase[1:5])
    season = phase[0]

    seasons = {"S": 0, "F": 1, "W": 1}
    return (year - 1901) * 2 + seasons[season]


def phase_to_phasetype(phase: Phase) -> str:
    phase_types = {"M": "Diplomacy", "R": "Retreats", "A": "Builds"}

    return phase_types[phase[5]]


def _build_coast_id_to_loc_id(loc_to_name: Dict[int, str]) -> Dict[int, int]:
    name2loc = {v: k for k, v in loc_to_name.items()}
    return {
        name2loc[coastal_name]: name2loc[coastal_name.split("/")[0]]
        for coastal_name in name2loc
        if "/" in coastal_name
    }


def get_requests_wrapper() -> requests.Session:
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES_FOR_NETWORK_ISSUES)  # type: ignore
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def get_req(url, params, api_key):
    logging.info(f"Hitting {params['route']}")
    resp = get_requests_wrapper().get(
        url, params=params, headers=get_api_header(api_key), timeout=60,
    )
    logging.info(f"Got response from {params['route']}")
    return resp


def post_req(url, params, json, api_key):
    logging.info(f"Hitting {params['route']}")
    resp = get_requests_wrapper().post(
        url, params=params, headers=get_api_header(api_key), json=json, timeout=60,
    )
    logging.info(f"Got response from {params['route']}")
    return resp


def get_status_json(ctx: Context) -> Optional[Json]:
    status_resp = get_req(
        ctx.api_url,
        {"route": STATUS_ROUTE, "gameID": ctx.gameID, "countryID": ctx.countryID},
        ctx.api_key,
    )

    if status_resp.content == GAME_NOT_FOUND_RESP:
        logging.warn(f"Game {ctx.gameID} disappeared.")
        raise WebdipGameNotFoundException

    if status_resp.content == ACCESS_DENIED_RESP:
        logging.warn(
            f"Game {ctx.gameID} gave access denied. This usually means the game was cancelled."
        )
        return None

    if status_resp.status_code != 200:
        logging.warn(
            f"Could not get status of {ctx.gameID}, country ID {ctx.countryID} for API key passed."
        )
        return None

    return safe_json_loads(status_resp.content)


def webdip_state_to_game(webdip_state_json: Json, stop_at_phase: Optional[Phase] = None) -> Game:
    if webdip_state_json["variantID"] == GameVariant.CLASSIC:
        game = Game()
    elif webdip_state_json["variantID"] == GameVariant.FVA:
        with (heyhi.PROJ_ROOT / "bin/game_france_austria.json").open() as stream:
            game = Game.from_json(stream.read())
    else:
        raise ValueError("Bad variant: %s" % webdip_state_json["variantID"])
    id_to_power = COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP[webdip_state_json["variantID"]]
    terr_id_to_loc = TERR_ID_TO_LOC_BY_MAP[webdip_state_json["variantID"]]

    # Set pot type
    pot_type = webdip_state_json["potType"]
    if pot_type == "Unranked" or pot_type == "Sum-of-squares":
        game.set_scoring_system(Game.SCORING_SOS)
    elif pot_type == "Points-per-supply-center":
        # We don't implement this scoring system, just use SOS which is similar
        # but squares it instead of being linear.
        game.set_scoring_system(Game.SCORING_SOS)
    elif pot_type == "Winner-takes-all":
        game.set_scoring_system(Game.SCORING_DSS)
    else:
        raise ValueError("Game has unknown pot_type: %s" % pot_type)

    # Set phase length
    phase_length = webdip_state_json["phaseLengthInMinutes"]
    game.set_metadata("phase_minutes", str(phase_length))

    # Webdip will sometimes send messages in the wrong phase -- e.g.
    # m["phaseMarker"] will be "Builds" but the message dict will be in the prior
    # "Diplomacy" phase. Here we gather all messages and annotate them with the
    # true phase, then search this list when actually adding messages to the Game
    all_messages: List[Dict[str, Any]] = [
        {**m, "phase": turn_to_phase(p["turn"], m["phaseMarker"])}
        for p in webdip_state_json["phases"]
        for m in p.get("messages", [])
        if m["fromCountryID"] != GLOBAL_CHAT_COUNTRY_ID and m["toCountryID"] != m["fromCountryID"]
    ]
    # Then delete the possibly erroneous messages in the "phases" dicts so we
    # don't accidentally access them
    for p in webdip_state_json["phases"]:
        if "messages" in p:
            del p["messages"]
    # Convert draw votes to messages
    all_messages.extend(
        [
            {
                **m,
                "phase": turn_to_phase(p["turn"], m["phaseMarker"]),
                "message": (
                    DRAW_VOTE_TOKEN
                    if m["vote"].lower() == DATASET_DRAW_MESSAGE.lower()
                    else UNDRAW_VOTE_TOKEN
                ),
                "fromCountryID": m["countryID"],
                "toCountryID": 0,  # ALL
            }
            for p in webdip_state_json["phases"]
            for m in p.get("publicVotesHistory", [])
            if (
                m.get("vote").lower()
                in [DATASET_DRAW_MESSAGE.lower(), DATASET_NODRAW_MESSAGE.lower()]
            )
        ]
    )
    # handle None phases returned by turn_to_phase
    inplace_handle_messages_with_none_phases(
        all_messages, webdip_state_json["turn"], webdip_state_json["phase"]
    )
    assert all(m["phase"] is not None for m in all_messages), all_messages

    def add_current_phase_messages() -> None:
        """Closure to facilitate adding messages"""
        for msg in [m for m in all_messages if m["phase"] == game.current_short_phase]:
            message = msg["message"]
            # fix webdip encoding of newlines
            message = message.replace("<br />", "\n")
            message = html.unescape(message)
            game.add_message(
                id_to_power[msg["fromCountryID"]],
                id_to_power[msg["toCountryID"]],
                message,
                time_sent=Timestamp.from_seconds(int(msg["timeSent"])),
                increment_on_collision=True,
            )

    for phase in webdip_state_json["phases"]:
        # an adj phase with no orders may not show up at all in the json: skip
        # ahead to the spring phase to stay sync'd
        if game.current_short_phase[-1] == "A" and phase["phase"] == "Diplomacy":
            logger.debug(f"Skip empty {game.phase}")
            game.process()

        # handle messages
        add_current_phase_messages()

        # handle orders
        terr_to_unit = {}
        for j in phase["units"]:
            if j["unitType"] == "":
                continue
            terr_to_unit[terr_id_to_loc[j["terrID"]]] = j["unitType"][0]

        orders = phase["orders"]
        power_to_orders = defaultdict(list)
        for order_json in phase["orders"]:
            if game.phase == stop_at_phase:
                break

            # 1. extract the data
            power = id_to_power[order_json["countryID"]]
            loc = terr_id_to_loc[order_json["terrID"]]
            from_loc = terr_id_to_loc[order_json["fromTerrID"]]
            to_loc = terr_id_to_loc[order_json["toTerrID"]]
            unit = order_json["unitType"][:1]
            if unit == "":
                unit = terr_to_unit.get(loc, "F")  # default to Fleet in case we missed a coast
            order_type = order_json["type"][0]
            if order_type == "M":
                order_type = "-"
            if order_type == "B":
                unit = order_json["type"].split()[1][0]  # e.g. type="Build Army"
            # if order_type == "D":

            via = "VIA" if order_json["viaConvoy"] == "Yes" else ""

            # 2. build the order string
            if from_loc != "":
                order_str = f"{unit} {loc} {order_type} {terr_to_unit.get(from_loc, 'F')} {from_loc} - {to_loc}"
            else:
                # note: default to Fleet in secondary location because sometimes
                # we get confused with NC / SC
                secondary_unit = terr_to_unit.get(to_loc, "F") + " " if order_type == "S" else ""
                order_str = f"{unit} {loc} {order_type} {secondary_unit}{to_loc} {via}".strip()

            possible_orders = game.get_all_possible_orders()[loc]
            # check if this is
            # a disband that has already been processed. This sometimes
            # happens when a unit is dislodged and has nowhere to
            # retreat -- there will be a disband in the db, but the
            # Game object disbands it automatically.
            if (
                phase["phase"] == "Retreats"
                and order_str.split()[-1] == "D"
                and (order_str not in possible_orders or not game.phase.endswith("RETREATS"))
            ):  # two cases: retreat phase with this order skipped; or retreat phase skipped entirely
                continue

            if order_str not in possible_orders:
                # if order is not valid, try location coastal variants, since
                # some orders are coming out of the db without the proper
                # coast.
                variant_split = get_valid_coastal_variant(order_str.split(), possible_orders)
                if variant_split is not None:
                    order_str = " ".join(variant_split)

            if order_str not in possible_orders:
                if is_duplicate_last_phase_orders(phase, webdip_state_json):
                    # we're done processing this game
                    return game

                # else, it's a bug
                assert order_str in possible_orders, (
                    game.phase,
                    (power, loc, order_str),
                    possible_orders,
                    order_json,
                )

            # logger.debug('set_phase_orders -> {} "{}"'.format(power, order_str))
            power_to_orders[power].append(order_str)

        for power, orders in power_to_orders.items():
            # logger.debug(f"Set {power} {orders}")
            game.set_orders(power, orders)

        if power_to_orders:
            game.process()

            # There is no extra completed phase in the status json, so this
            # loop will naturally exit before adding post-game messages. Here
            # we explicitly check for this (and assert for correctness below by
            # checking the added message counts).
            if game.current_short_phase == "COMPLETED":
                add_current_phase_messages()
                break

    # Sanity check that we haven't missed any messages
    added_messages = [m for p in game.get_all_phases() for m in p.messages]
    with GLOBAL_SLACK_EXCEPTION_SWALLOWER:
        assert len(all_messages) == len(added_messages), (len(all_messages), len(added_messages))

    return game


def inplace_handle_messages_with_none_phases(
    messages: List[Dict], turn: int, status_json_phase: str
) -> None:
    """Handle None phases returned by turn_to_phase

    - None-phase messages with phaseMarker "Finished" and game finished -> "COMPLETED"
    - None-phase messages at the beginning of the game -> S1901M
    - None-phase messages after a non-None-phase messages -> Exception
    """
    if (
        len(messages) > 1
        and turn > 1  # at least 1902
        and all(m["phase"] is None for m in messages)
        and any(m["fromCountryID"] != 0 and m["toCountryID"] != 0 for m in messages)
    ):
        raise UnexpectedWebdipBehaviorException(f"All None-phase messages: {messages}")
    good_phase_encountered = False
    for m in messages:
        if m["phase"] is None:
            if status_json_phase == m.get("phaseMarker") == "Finished":
                m["phase"] = "COMPLETED"
            elif good_phase_encountered:
                raise UnexpectedWebdipBehaviorException(f"Bad None-phase message: {m}")
            else:
                m["phase"] = "S1901M"
        else:
            good_phase_encountered = True


def is_duplicate_last_phase_orders(phase: Json, webdip_state_json: Json):
    if phase["phase"] != "Diplomacy" or phase["turn"] != webdip_state_json["phases"][-1]["turn"]:
        return False
    prev_move_phase = next(
        p
        for p in webdip_state_json["phases"]
        if p["phase"] == "Diplomacy" and p["turn"] == (phase["turn"] - 1)
    )
    prev_move_orders = set(strip_key_and_freeze(d, "turn") for d in prev_move_phase["orders"])
    this_move_orders = set(strip_key_and_freeze(d, "turn") for d in phase["orders"])
    return prev_move_orders == this_move_orders


def strip_key_and_freeze(d, key):
    d = {k: v for k, v in d.items() if k != key}
    return frozenset(d.items())


class WrappedJSONDecodeError(RuntimeError):
    pass


def safe_json_loads(json_str: Union[bytes, str]) -> Json:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise WrappedJSONDecodeError(f"Bad JSON: {json_str}") from e


def str_timedelta(d: timedelta):
    # str(d) gives wacky results when d < 0
    if d < timedelta(seconds=0):
        return "-" + str(-d)
    else:
        return str(d)


def get_api_header(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def construct_game_fp(
    game_dir: pathlib.Path, ctx: Context, id_to_power: Dict[int, str]
) -> pathlib.Path:
    return game_dir / f"game_{ctx.gameID}_{id_to_power[ctx.countryID]}.json"


def get_default_checkpoint_path(
    basedir: Union[str, pathlib.Path], api_url: str, api_key: str
) -> pathlib.Path:
    url_hash = hashlib.md5(api_url.encode("UTF-8")).hexdigest()[:6]
    return pathlib.Path(os.path.join(basedir, f"{url_hash}__{api_key}.pt"))


def retry_on_connection_error(func):
    def wrapped(*args, **kwargs):
        N = len(RECONNECT_SLEEP_TIMES)
        for try_idx, retry_time in enumerate(RECONNECT_SLEEP_TIMES + [-1]):
            try:
                res = func(*args, **kwargs)
                if try_idx > 0:
                    logging.info(f"Retry {try_idx} was successful.")
                return res
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                # webdip uses die(...) for a lot of errors, which returns
                # a successful response code with the error message so we end up catching
                # it as a JSONDecodeError.
                WrappedJSONDecodeError,
            ) as err:
                if retry_time > 0:
                    logging.info(traceback.format_exc())
                    logging.info(
                        f"Connection error. Retrying in {retry_time} seconds. Will try {N - try_idx} more times."
                    )
                    time.sleep(retry_time)
                    logging.info(f"Retrying now (try {try_idx + 2})")
                else:
                    raise err

        assert False

    return wrapped


def get_agent_draw_vote(status_json: Json) -> bool:
    """Returns True if this agent is currently voting for a draw."""
    # Just in case
    if "votes" not in status_json:
        logging.warning(f"status_json is missing votes field: {status_json}")
        return False
    if status_json["votes"] is None:
        return False
    votes = str(status_json["votes"]).split(",")
    logging.info(f"Votes: {votes}")
    return "Draw" in votes


@retry_on_connection_error
def set_draw_vote(ctx: Context, desired_vote: bool) -> Optional[Json]:
    """Set the agent's draw voting status to desired_vote. Returns the new status_json on success, None on failure."""
    # set_draw_vote wraps getting status_json within it rather than taking it as an argument
    # because upon connnection failure, webdip may or may not have actually performed the toggle.
    # We can't blindly retry since we don't know what state the toggle is in, so we need to
    # get the state fresh again each time.

    try:
        status_json = get_status_json(ctx)
    except WebdipGameNotFoundException:
        logging.error("set_draw_vote: could not get status json, game not found")
        return None
    if status_json is None:
        logging.error("set_draw_vote: could not get status json, got None")
        return None

    if get_agent_draw_vote(status_json) == desired_vote:
        logging.info(f"set_draw_vote: draw vote is already {desired_vote}, not changing it")
        return status_json

    toggle_vote_json = {
        "gameID": ctx.gameID,
        "countryID": ctx.countryID,
        "vote": "Draw",
    }

    logging.info(f"JSON: {pformat(toggle_vote_json)}")
    resp = post_req(ctx.api_url, {"route": VOTE_ROUTE, **toggle_vote_json}, None, ctx.api_key,)

    if resp.status_code != 200:
        logging.error("Could not toggle vote: %s", resp.content.decode())
        return None
    logging.info(f"set_draw_vote: draw vote is now set to {desired_vote}")

    # wait until draw vote has been recorded in the message history by counting
    # the number of outbound draw votes before and after toggling (count
    # *outbound* because someone else may draw vote concurrently)
    def count_votes_by(_status_json: Json, vote_str: str, countryID: int) -> int:
        return len(
            [
                vote
                for phase in _status_json["phases"]
                for vote in phase.get("publicVotesHistory", [])
                if vote["countryID"] == countryID and vote["vote"].lower() == vote_str.lower()
            ]
        )

    vote_str = DATASET_DRAW_MESSAGE if desired_vote else DATASET_NODRAW_MESSAGE
    n_votes_before = count_votes_by(status_json, vote_str, ctx.countryID)
    status_json = poll_until(
        ctx, lambda _j: count_votes_by(_j, vote_str, ctx.countryID) == (n_votes_before + 1)
    )
    if status_json is None:
        raise UnexpectedWebdipBehaviorException(
            f"draw vote toggled but never appeared in publicVotesHistory: {ctx} {desired_vote}"
        )
    return status_json


def poll_until(
    ctx: Context,
    fn: Callable[[Json], bool],
    *,
    retry_every_n_seconds: float = 1,
    retry_n_times: int = 10,
) -> Optional[Json]:
    """Get status json on repeat until condition is True"""
    for i in range(retry_n_times):
        status_json = get_status_json(ctx)
        if status_json and fn(status_json):
            return status_json
        time.sleep(retry_every_n_seconds)
    return None


def poll_until_message_appears(ctx: Context, timestamp: Timestamp) -> Optional[Json]:
    """Get status json on repeat until message appears with timestamp. Return json or None."""
    timestamp_seconds = timestamp.to_seconds_int()
    return poll_until(
        ctx,
        lambda status_json: any(
            [
                m["timeSent"] == timestamp_seconds
                for phase in reversed(status_json["phases"])
                for m in phase.get("messages", [])
            ]
        ),
    )


def get_keep_alive_and_last_restart_keys(bot_account_name: str,) -> Tuple[str, str]:
    prefix = "webdip-infra"
    keep_alive_key = f"{prefix}:{bot_account_name}:keep-alive"
    last_restart_key = f"{prefix}:{bot_account_name}:last-restart"
    return keep_alive_key, last_restart_key


def get_keep_alive_and_last_restart_ts(
    bot_account_name: str,
) -> Tuple[Optional[Timestamp], Optional[Timestamp]]:
    keep_alive_key, last_restart_key = get_keep_alive_and_last_restart_keys(bot_account_name)
    redis_host = get_redis_host()

    keep_alive: Optional[Timestamp] = redis_host.get(keep_alive_key)
    if keep_alive:
        keep_alive = Timestamp.from_centis(keep_alive)

    last_restart: Optional[Timestamp] = redis_host.get(last_restart_key)
    if last_restart:
        last_restart = Timestamp.from_centis(last_restart)

    return keep_alive, last_restart


def set_keep_alive_and_last_restart_ts(
    bot_account_name: str,
    keep_alive: Optional[Timestamp] = None,
    last_restart: Optional[Timestamp] = None,
):
    keep_alive_key, last_restart_key = get_keep_alive_and_last_restart_keys(bot_account_name)
    redis_host = get_redis_host()

    if keep_alive:
        assert redis_host.set(keep_alive_key, keep_alive)

    if last_restart:
        assert redis_host.set(last_restart_key, last_restart)


def get_is_new_msgs_from_sender_since_ts(game: Game, sender: Power, last_ts: Timestamp) -> bool:

    ts_to_msg_dcts = {
        **{
            ts: msg_dct
            for phase_data in game.message_history.values()
            for ts, msg_dct in phase_data.items()
        },
        **game.messages,
    }
    return any(
        ts > last_ts and msg_dct["sender"] == sender for ts, msg_dct in ts_to_msg_dcts.items()
    )


def get_message_history_key(ctx: Context) -> str:
    # We do not want to use game_fp here because each recipient worker will have
    # different game_fp and we want to share this info accross all workers for a
    # game. The records in the Redis have TTL that is less than bot loading time
    # that makes any issues unlikely.
    ctx_verbose = ctx._replace(countryID=PRESS_COUNTRY_ID_TO_POWER[ctx.countryID])
    return "|".join(map(str, ctx_verbose))


class WebdipBotWrapper:
    SERIALIZED_ATTRS = (
        "context_to_dialogue_state",
        "context_last_checked",
        "dialogue_wakeup_times",
        "annotators",
    )

    def __init__(
        self,
        cfg: conf_cfgs.PlayWebdipTask,
        agent: BaseAgent,
        api_url: str,
        api_key: str,
        account_name: str,
        game_ids: Optional[List[int]],
        game_name: Optional[str],
        game_dir: pathlib.Path,
        checkpoint_dir: pathlib.Path,
        sleep_multiplier: float,
        expected_recipient: Optional[Power],
    ):

        self.cfg = cfg
        self.agent = agent
        # Maps game id -> player for that game. Each player shares the same agent
        # configuration (self.agent) but tracks its own per-power state for that game.
        self.players: Dict[Context, Player] = {}
        self.api_url = api_url
        self.api_key = api_key
        self.account_name = account_name
        self.game_ids = game_ids
        self.game_name = game_name
        self.context_to_dialogue_state: Dict[Context, DialogueState] = {}
        # this keeps track of contexts we've already serviced, so that we services
        # games in a fair round-robin fashion, avoiding starvation
        # Element 0 is the next to be serviced, and element N is the most recently serviced
        self.context_last_checked: Dict[Context, float] = {}
        self.dialogue_wakeup_times: Dict[Context, Timestamp] = {}
        self.estimated_msg_generation_time: Timestamp = Timestamp.from_seconds(0)
        self.game_dir = game_dir
        self.checkpoint_dir = checkpoint_dir
        self.annotators: Dict[Context, MetaAnnotator] = {}
        self.sleep_multiplier = sleep_multiplier
        self.latest_status_json: Optional[Json] = None
        assert cfg.variant_id in list(GameVariant), f"Unknown game variant {cfg.variant_id}"

        message_handler: Optional[ParlaiMessageHandler] = getattr(
            self.agent, "message_handler", None
        )
        if message_handler is not None:
            message_handler.model_dialogue.set_block_redacted_tokens()
            assert not (
                message_handler.use_pseudoorders_initiate_sleep_heuristic
                and expected_recipient is None
            ), "If use_pseudoorders_initiate_sleep_heuristic is True we need a specificied recipient"

        self.last_successful_message_time: Dict[Context, Timestamp] = {}

    def purge_ctx(self, ctx: Context, reason: str = "no reason provided"):
        logging.info(f"Purging context {ctx} with reason: {reason}")
        if ctx in self.context_last_checked:
            del self.context_last_checked[ctx]
        if ctx in self.context_to_dialogue_state:
            del self.context_to_dialogue_state[ctx]
        if ctx in self.dialogue_wakeup_times:
            del self.dialogue_wakeup_times[ctx]

        game_fp = str(construct_game_fp(self.game_dir, ctx, PRESS_COUNTRY_ID_TO_POWER))
        delete_message_review(game_fp)

    def _get_uid(self):
        return (self.api_url, self.api_key)

    def maybe_load_from_checkpoint(self, path: pathlib.Path) -> None:
        if not os.path.exists(path):
            logging.info(f"Did not find a checkpoint at {path}")
            return

        data = torch.load(path)
        logging.info(f"Loading bot state from checkpoint at {path}:")
        logging.info(f"{pformat(data)}")
        for k, v in data.items():
            setattr(self, k, v)

        if (
            "CHECKPOINT_CFG" in data
            and data["CHECKPOINT_CFG"]["agent"] != self.cfg.to_dict()["agent"]
        ):
            logging.warning("cfg has changed from checkpoint. Will not load state dicts.")
            logging.warning(f"Old cfg:\n{data['CHECKPOINT_CFG']}")
            logging.warning(f"New cfg:\n{self.cfg.to_dict()}")
            return

        if "PLAYER_STATE_DICTS" in data:
            for ctx, state_dict in data["PLAYER_STATE_DICTS"].items():
                if ctx not in self.players:
                    # relies on "power" being the name of the key in Player's state_dict
                    self.players[ctx] = Player(self.agent, state_dict["power"])
                self.players[ctx].load_state_dict(state_dict)

    def maybe_save_checkpoint(self, path: pathlib.Path) -> None:
        data = {a: getattr(self, a) for a in self.SERIALIZED_ATTRS}
        data["PLAYER_STATE_DICTS"] = {
            ctx: player.state_dict() for ctx, player in self.players.items()
        }
        data["CHECKPOINT_CFG"] = self.cfg.to_dict()
        logging.info(f"Saving agent/webdip state checkpoint to {path.absolute()}")
        with atomicish_open_for_writing_binary(path) as f:
            torch.save(data, f)

    def post_process(self, ctx: Context):
        logging.info("Running bot post-processing (e.g. checkpoint state)")

        if meta_annotations.has_annotator():
            cur_annotator = meta_annotations.pop_annotator()
            self.annotators[ctx] = cur_annotator

        self.maybe_save_checkpoint(self.checkpoint_dir)

    @retry_on_connection_error
    def get_missing_orders_json(self):
        missing_orders_resp = get_req(self.api_url, {"route": MISSING_ORDERS_ROUTE}, self.api_key)
        return safe_json_loads(missing_orders_resp.content)

    @retry_on_connection_error
    def get_active_games_json(self):
        active_games_resp = get_req(self.api_url, {"route": ACTIVE_GAMES_ROUTE}, self.api_key)
        resp = safe_json_loads(active_games_resp.content)
        if "games" not in resp:
            logging.error(f"Bad active games response: {resp}")
            return []
        return resp["games"]

    def make_context_from_json(self, j):
        return Context(
            gameID=j["gameID"],
            countryID=j["countryID"],
            api_url=self.api_url,
            api_key=self.api_key,
        )

    def find_context(self) -> Tuple[Optional[Context], bool]:
        cfg = self.cfg
        cur_ctx: Optional[Context] = None
        status_json = None

        if cfg.check_phase or cfg.force or cfg.json_out:
            # Note, this assumes default map.
            assert (
                self.game_ids is not None and len(self.game_ids) == 1
            ), "Must provide one game id in check_phase, force, and json_out modes"
            cur_ctx = Context(
                self.game_ids[0], COUNTRY_POWER_TO_ID[cfg.force_power], self.api_url, self.api_key,
            )
            return cur_ctx, False

        active_games_json = self.get_active_games_json()
        logger.info(f"All active games: {pformat(active_games_json)}")

        # Filter games by game_id (list of int game ids) or game_name (wildcard string-matched game name)
        logger.info(f"Filtering games, game_ids= {self.game_ids}, game_name= {self.game_name}")
        active_games_json_filtered = [
            x
            for x in active_games_json
            if (self.game_ids is None or x["gameID"] in self.game_ids)
            and (self.game_name is None or fnmatch(x["name"], self.game_name))
        ]

        if len(active_games_json_filtered) != len(active_games_json):
            logger.info(f"Filtered games: {pformat(active_games_json_filtered)}")

        missing_orders_json_filtered = [
            x
            for x in active_games_json_filtered
            if "Saved" not in x["orderStatus"].split(",")
            and "Ready" not in x["orderStatus"].split(",")
            and "None" not in x["orderStatus"].split(",")
        ]
        logger.info(f"All games awaiting orders: {pformat(missing_orders_json_filtered)}")

        if len(missing_orders_json_filtered) > 0:
            found_ctx = self.make_context_from_json(missing_orders_json_filtered[0])
            if self.cfg.allow_dialogue:
                game_fp = str(
                    construct_game_fp(self.game_dir, found_ctx, PRESS_COUNTRY_ID_TO_POWER)
                )
                self.get_message_review_and_update_last_serviced(game_fp)
            return found_ctx, True

        # service games that may be waiting on messages
        press_ctxs = [
            self.make_context_from_json(j)
            for j in active_games_json_filtered
            if j["pressType"] != "NoPress"
        ]

        # service contexts in round robin order
        press_ctxs_sorted = sorted(
            press_ctxs, key=lambda ctx: self.context_last_checked.get(ctx, -math.inf)
        )
        logging.info(f"press_ctxs_sorted= {press_ctxs_sorted}")
        logging.info(f"ctxs_to_game_states= {self.context_to_dialogue_state}")
        logging.info(f"context_last_checked: {self.context_last_checked}")
        for ctx in press_ctxs_sorted:
            if self.game_ids is not None and ctx.gameID not in self.game_ids:
                continue
            self.context_last_checked[ctx] = time.time()

            try:
                status_json = self.get_status_json(ctx)
            except WebdipGameNotFoundException:
                self.purge_ctx(ctx, "Webdip game not found.")
            if status_json is None:
                # Sometimes, webdip has availability issues
                logging.info("Could not get status json!")
                continue

            cur_game = webdip_state_to_game(status_json, stop_at_phase=cfg.check_phase)
            if self.cfg.recipient and self.cfg.recipient not in cur_game.get_alive_powers():
                continue
            if not cfg.check_phase:
                update_phase_message_history_state(
                    game_fp=get_message_history_key(ctx),
                    game=cur_game,
                    agent_power=PRESS_COUNTRY_ID_TO_POWER[ctx.countryID],
                )

            logging.info(
                f"for ctx={ctx}, realtime state is {cur_game.phase, self.get_message_history_length(ctx, cur_game)}"
            )

            if (
                status_json["gameOver"] != "No"
                or PRESS_COUNTRY_ID_TO_POWER[ctx.countryID] not in cur_game.get_alive_powers()
            ):
                reason = ""
                if status_json["gameOver"] != "No":
                    reason += "Game over. "
                if PRESS_COUNTRY_ID_TO_POWER[ctx.countryID] not in cur_game.get_alive_powers():
                    reason += "Player died."
                self.purge_ctx(ctx, reason)

            if self.check_for_actionable_message_state(ctx, cur_game):
                return ctx, False

        return None, False

    @retry_on_connection_error
    def get_status_json(self, ctx: Context) -> Optional[Json]:
        logging.info(f"get_status_json for {ctx}")
        status_json = get_status_json(ctx)
        self.latest_status_json = status_json
        return status_json

    @retry_on_connection_error
    def maybe_resend_orders(self, ctx: Context, status_json: Json) -> bool:
        """Catch corner case where we need to re-send orders from last phase.

        Returns true if this case is hit.
        """

        if not self.cfg.check_phase and status_json["phases"]:
            last_phase_json = status_json["phases"][-1]
            if (
                status_json["turn"] == last_phase_json["turn"]
                and status_json["phase"] == last_phase_json["phase"]
                and last_phase_json["orders"]
            ):
                shuttle_path = pathlib.Path("shuttle.%s.pt" % datetime.now().isoformat()).resolve()
                shuttle = dict(status_json=status_json)
                torch.save(shuttle, shuttle_path)
                logging.warning(
                    "Status JSON already contains orders for last phase. Will try to do something. See all data in\n%s",
                    shuttle_path,
                )

                if status_json["phase"] == "Builds":
                    logging.info("Hit a builds phase (when we shouldn't have?)")
                    return True
        return False

    def has_state_changed(self, ctx: Context, game: Game):
        prev_state = self.context_to_dialogue_state.get(ctx)
        cur_state = (game.phase, self.get_message_history_length(ctx, game))

        game_fp = str(construct_game_fp(self.game_dir, ctx, PRESS_COUNTRY_ID_TO_POWER))
        message_review = get_message_review(game_fp)

        logging.info(
            f"prev_state={prev_state}, "
            f"cur_state={cur_state}, "
            f"has_message_review={message_review is not None}, "
            f"game.current_short_phase={game.current_short_phase}, "
            f"message_review['short_phase']={message_review['short_phase'] if message_review else None}"
        )

        # In 5 min games, only update message review on incoming message if incoming message sender is same as recipient of current message proposal
        if (
            prev_state
            and message_review
            and self.cfg.recipient is not None
            and self.cfg.only_bump_msg_reviews_for_same_power
            and len(message_review["msg_proposals"]) > 0
        ):
            return prev_state[0] != cur_state[0] or get_is_new_msgs_from_sender_since_ts(
                game, self.cfg.recipient, message_review["last_timestamp_when_produced"],
            )

        return cur_state != prev_state

    def has_phase_changed(self, ctx: Context, game: Game):
        prev_state = self.context_to_dialogue_state.get(ctx)
        cur_phase = game.phase
        return not prev_state or prev_state[0] != cur_phase

    def check_wakeup(self, ctx: Context):
        wakeup_time = self.dialogue_wakeup_times.get(ctx)
        if wakeup_time is None:
            return False
        time_until_wakeup = max(Timestamp.from_seconds(0), wakeup_time - Timestamp.now())
        logging.info(
            f"{ctx} will wakeup in {time_until_wakeup} ; reserving {self.estimated_msg_generation_time} for message generation"
        )
        return time_until_wakeup <= self.estimated_msg_generation_time

    def get_message_history_length(self, ctx: Context, game: Game,) -> int:
        all_timestamps = [
            t
            for phase in game_from_view_of(
                game, PRESS_COUNTRY_ID_TO_POWER[ctx.countryID]
            ).get_all_phases()
            for t in phase.messages.keys()
        ]

        return len(all_timestamps)

    def botgame_fp_to_context(self, game_fp: str) -> Context:
        return botgame_fp_to_context(game_fp)._replace(api_url=self.api_url, api_key=self.api_key)

    def get_last_timestamp_or_zero(self, game_fp: str, game: Game) -> Timestamp:
        """
        Looks for most recent message in this game (i.e. all phases including current one) and returns its timestamp, returning 0 otherwise
        """
        ctx = self.botgame_fp_to_context(game_fp)

        all_timestamps = [
            t
            for phase in game_from_view_of(
                game, PRESS_COUNTRY_ID_TO_POWER[ctx.countryID]
            ).get_all_phases()
            for t in phase.messages.keys()
        ]

        return max(all_timestamps) if len(all_timestamps) > 0 else Timestamp.from_seconds(0)

    def get_last_timestamp_this_phase(
        self, game: Game, default: Timestamp = Timestamp.from_seconds(0)
    ) -> Timestamp:
        """
        Looks for most recent message in this phase and returns its timestamp, returning default otherwise
        """
        all_timestamps = game.messages.keys()
        return max(all_timestamps) if len(all_timestamps) > 0 else default

    def reuse_stale_pseudo(self, ctx):
        last_msg_time = self.last_successful_message_time.get(ctx)
        if self.cfg.reuse_stale_pseudo_after_n_seconds >= 0 and last_msg_time:
            delta = Timestamp.now() - last_msg_time
            logging.info(f"reuse_stale_pseudo: delta= {delta / 100:.2f} s")
            return delta > Timestamp.from_seconds(self.cfg.reuse_stale_pseudo_after_n_seconds)
        else:
            return False

    def get_should_stop_condition(
        self, *, initial_game: Game, game_fp: str, recipient: Optional[Power] = None
    ) -> Callable[[bool], bool]:
        if recipient is None and self.cfg.only_bump_msg_reviews_for_same_power:
            # Can't filter for the recipient if we don't know who they are.
            return lambda x: False

        ctx = self.botgame_fp_to_context(game_fp)
        agent_power = PRESS_COUNTRY_ID_TO_POWER[ctx.countryID]

        the_power = recipient if self.cfg.only_bump_msg_reviews_for_same_power else agent_power
        assert the_power is not None

        original_state = compute_phase_message_history_state_with_power(initial_game, agent_power)[
            the_power
        ]

        def should_stop(post_pseudoorders: bool) -> bool:
            try:
                status_json = self.get_status_json(ctx)
            except WebdipGameNotFoundException:
                logging.warning("Webdip game not found")
                return False
            if status_json is None:
                logging.info("Could not get status json!")
                return False
            cur_game = webdip_state_to_game(status_json)
            maybe_power_state = compute_phase_message_history_state_with_power(
                cur_game, agent_power
            )
            if maybe_power_state is None:
                return False
            new_state = maybe_power_state.get(the_power)
            if new_state is None:
                return False

            is_outdated = new_state > original_state

            # if it's been more than reuse_stale_pseudo_after_n_seconds seconds since
            # we last generated a message, we will not stop message generation until
            # pseudo-orders are completed
            use_stale_pseudo = self.reuse_stale_pseudo(ctx)

            logging.info(
                "Should stop check: power=%s start_state=%s current_state=%s is_outdated=%s post_pseudo=%s use_stale=%s",
                the_power,
                original_state,
                new_state,
                is_outdated,
                post_pseudoorders,
                use_stale_pseudo,
            )
            return is_outdated and (post_pseudoorders or not use_stale_pseudo)

        return should_stop

    def generate_message_for_approval(
        self, game_fp: str, game: Game, recipient: Optional[Power]
    ) -> bool:
        """
        1) remove entries for game in context from cache
        2) generate new message
        3) push new message to cache
        """
        # Add any notes about generation.
        comment = ""

        ctx = self.botgame_fp_to_context(game_fp)
        player = self.players[ctx]
        status_json = self.get_status_json(ctx)
        assert status_json

        if ctx not in self.last_successful_message_time:
            self.last_successful_message_time[ctx] = Timestamp.now()

        # Get basis for elapsed time
        # - last message timestamp this phase
        # - if this is the first message this phase, use current system time to approximate phase transition time
        # Add sleep time to get message timestamp "wakeup_time"
        last_timestamp_this_phase = self.get_last_timestamp_this_phase(
            game, default=Timestamp.now()
        )
        sleep_time = player.get_sleep_time(game, recipient)
        wakeup_time = last_timestamp_this_phase + sleep_time

        # Keep model in-distribution by not actually conditioning on inf
        # timestamp, used for force-sending
        sleep_time_for_conditioning = (
            sleep_time if sleep_time < INF_SLEEP_TIME else MESSAGE_DELAY_IF_SLEEP_INF
        )

        # Special case: since we don't know timestamp of game start,
        # generate_message expects sleep_time as the first timestamp
        if get_last_message(game) is None:
            timestamp_for_conditioning = sleep_time_for_conditioning
        else:
            timestamp_for_conditioning = last_timestamp_this_phase + sleep_time_for_conditioning

        msgs: List[MessageDict] = []
        should_stop = self.get_should_stop_condition(
            initial_game=game, game_fp=game_fp, recipient=recipient
        )
        for _ in range(3):
            # Temporarily pause annotator when generating additional proposals
            annotator = meta_annotations.pop_annotator() if len(msgs) > 0 else None

            pseudo_orders = None
            if self.reuse_stale_pseudo(ctx) and isinstance(player.state, SearchBotAgentState):
                pseudo_orders = player.state.pseudo_orders_cache.maybe_get(
                    game, player.power, True, True, recipient
                )
            try:
                with set_interruption_condition(should_stop):
                    msg = player.generate_message(
                        game,
                        recipient=recipient,
                        timestamp=timestamp_for_conditioning,
                        pseudo_orders=pseudo_orders,
                    )

            except ShouldStopException:
                if annotator is None:
                    meta_annotations.after_message_generation_failed()
                raise
            finally:
                if annotator is not None:
                    meta_annotations.push_annotator(annotator)

            # Immediately quit out on the first failure to generate a message.
            # This is so that the game's phase_minutes or whether it requires message_approval
            # don't fundamentally change the number of tries we get at generating a sensical message
            # before we give up and set sleep time to infinity for future tries.
            if not msg:
                break

            msgs.append(msg)
            if not self.cfg.require_message_approval or game.get_metadata("phase_minutes") == "5":
                # No need to collect additional messages if no annotations happening or in 5min game.
                break

        if len(msgs) == 0:
            wakeup_time = last_timestamp_this_phase + INF_SLEEP_TIME
            msgs.append(
                build_message_dict(
                    player.power, player.power, "", game.current_short_phase, wakeup_time,
                )
            )
            comment += "Player:generate_message did not return a message! Returning blank message as placeholder."
            logging.error(
                "Player:generate_message did not return a message! Possibly, the proposed message was removed by the message filter. Adding blank message to self."
            )
        else:
            message_heuristic_result = player.agent.postprocess_sleep_heuristics_should_trigger(
                msgs[0], game, player.state
            )
            if message_heuristic_result == MessageHeuristicResult.FORCE:
                logging.info(f"Postprocess sleep heuristics triggered FORCE for {msgs[0]}")
                wakeup_time = timestamp_for_conditioning
            elif message_heuristic_result == MessageHeuristicResult.BLOCK:
                logging.info(f"Postprocess sleep heuristics triggered BLOCK for {msgs[0]}")
                wakeup_time = INF_SLEEP_TIME
            assert wakeup_time is not None
            logging.info(
                "The message generated (wake in %ds from now): %s",
                wakeup_time.to_seconds_int() - Timestamp.now().to_seconds_int(),
                msgs[0],
            )

        game_url = f"http://localhost:8894/message_review?cur_game={game_fp}"
        if "webdiplomacy.net" in self.cfg.webdip_url:
            send_slack_message(
                "message-review-notifications",
                f"New message to review for game ID {ctx.gameID}, country {PRESS_COUNTRY_ID_TO_POWER[ctx.countryID]} <{game_url}|here>.",
            )

        message_review_data: MessageReviewData = {
            "id": gen_id(),
            "game_id": ctx.gameID,
            "power": PRESS_COUNTRY_ID_TO_POWER[ctx.countryID],
            "msg_proposals": [
                {
                    "target_power": msg["recipient"],
                    "msg": msg["message"],
                    "approval_status": ApprovalStatus.UNREVIEWED,
                    "tags": [],
                }
                for msg in msgs
            ],
            "wakeup_time": wakeup_time,
            "last_timestamp_when_produced": self.get_last_timestamp_or_zero(game_fp, game),
            "last_serviced": Timestamp.now(),
            "user": "",
            "comment": comment,
            "annotator": (
                meta_annotations.get_annotator() if meta_annotations.has_annotator() else None
            ),
            "cfg": json.dumps(self.cfg.to_dict()),
            "phase_end_timestamp": Timestamp.from_seconds(int(status_json["processTime"] or 0)),
            "short_phase": game.current_short_phase,
            "parallel_recipient": recipient,
            "flag_as_stale": False,
        }

        if status_json["processTime"] is None:
            logging.error('ERROR: status_json["processTime"] is None. Status:\n%s', status_json)

        self.dialogue_wakeup_times[ctx] = wakeup_time
        logging.info(
            f"New wakeup time ({self.dialogue_wakeup_times[ctx].to_seconds_int()} seconds)"
        )

        # if sleep time is not inf, we don't want to update the message time until an action is taken
        if sleep_time == INF_SLEEP_TIME:
            self.last_successful_message_time[ctx] = Timestamp.now()

        set_message_review(
            game_fp, message_review_data,
        )

        return True

    def get_message_review_and_update_last_serviced(
        self, game_fp: str
    ) -> Optional[MessageReviewData]:
        message_review = get_message_review(game_fp)
        if not message_review:
            return None
        message_review["last_serviced"] = Timestamp.now()
        set_message_review(game_fp, message_review)
        return message_review

    def check_for_actionable_message_state(self, ctx: Context, cur_game: Game) -> bool:
        # Don't process for dead recipients when in parallel recipient mode
        if self.cfg.recipient and self.cfg.recipient not in cur_game.get_alive_powers():
            logging.info(
                f"Parallel recipient '{self.cfg.recipient}' is not in the list of alive powers for this game: {cur_game.get_alive_powers()}. Skipping servicing this recipient."
            )
            return False

        # Recompute messages when state has changed.
        if self.has_state_changed(ctx, cur_game):
            return True

        game_fp = str(construct_game_fp(self.game_dir, ctx, PRESS_COUNTRY_ID_TO_POWER))
        message_review = self.get_message_review_and_update_last_serviced(game_fp)

        if not message_review:
            # Need to regenerate a message review
            return True

        if message_review["short_phase"] != cur_game.current_short_phase:
            logging.info(
                f"Found phase mismatch between current message proposal's phase and current game phase: {message_review['short_phase'], cur_game.current_short_phase}"
            )
            return True

        if not self.cfg.only_bump_msg_reviews_for_same_power:
            if (
                self.get_last_timestamp_or_zero(game_fp, cur_game)
                != message_review["last_timestamp_when_produced"]
            ):
                return True

        # Re-gen when proposal marked as stale
        if message_review["flag_as_stale"]:
            logging.info("Found stale message proposal!")
            return True

        # Attempt to re-gen when previous message generation attempt failed
        if (
            message_review["msg_proposals"][0]["msg"] == ""
            and message_review["power"] == message_review["msg_proposals"][0]["target_power"]
        ):
            logging.info("Failed to generate a message; will not retry until state change.")
            return False

        if message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.REJECTED:
            # Need to regenerate a message review when a message is rejected
            return True
        elif self.check_wakeup(ctx) and (
            message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.APPROVED
            or (
                not self.cfg.require_message_approval
                and message_review["msg_proposals"][0]["target_power"] != message_review["power"]
            )
        ):
            # Need to send message when bot has woken up and message has been approved
            return True
        elif message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.FORCE_SEND:
            return True
        else:
            return False

    @retry_on_connection_error
    def send_message(
        self, ctx: Context, id_to_power: Dict[int, Power], msg: OutboundMessageDict
    ) -> Tuple[Json, Timestamp]:
        if get_kill_switch():
            raise WebdipSendMessageFailedException(f"Kill switch pulled, refusing to send message")
        power = id_to_power[ctx.countryID]
        power_to_id = {v: k for k, v in id_to_power.items()}
        logging.info(f"send_message {power} --> {msg['recipient']}: {msg['message']}")
        # if this is a draw vote, use a different API
        if msg["message"] in (DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN):
            new_status_json = set_draw_vote(ctx, msg["message"] == DRAW_VOTE_TOKEN)
            if new_status_json is None:
                raise WebdipSendMessageFailedException("failed to send draw vote")
            # read timestamp from status json
            last_draw_vote = [
                v
                for v in new_status_json["phases"][-1]["publicVotesHistory"]
                if v["countryID"] == ctx.countryID
            ][-1]
            return new_status_json, Timestamp.from_seconds(last_draw_vote["timeSent"])

        # otherwise this is a normal message, not a vote
        msg_json = {
            "gameID": ctx.gameID,
            "countryID": ctx.countryID,
            "toCountryID": power_to_id[msg["recipient"]],
            "message": msg["message"],
        }

        assert msg_json["message"] not in [DRAW_VOTE_TOKEN, UNDRAW_VOTE_TOKEN], msg_json
        send_message_resp = post_req(
            self.api_url, {"route": SEND_MESSAGE_ROUTE}, msg_json, self.api_key,
        )

        if send_message_resp.status_code != 200:
            logging.error(f"Message send failed: {send_message_resp.content}")
            raise WebdipSendMessageFailedException(
                f"send_message request failed: {send_message_resp.status_code}"
            )
        time_sent = safe_json_loads(send_message_resp.content)
        logging.info(f"time_sent= {time_sent}")
        if not isinstance(time_sent, int):
            # new server sends back the whole message object
            resp_msgs = time_sent["messages"]
            assert len(resp_msgs) == 1
            time_sent = resp_msgs[0]["timeSent"]
        timestamp = Timestamp.from_seconds(time_sent)
        logging.info(
            "Timestamp mismatch: (now - message_sent) = %d",
            Timestamp.now().to_seconds_int() - timestamp.to_seconds_int(),
        )
        status_json = poll_until_message_appears(ctx, timestamp)
        if status_json is None:
            raise WebdipSendMessageFailedException(
                f"message with timestamp {timestamp} never appeared"
            )
        return status_json, timestamp

    def run_message_approval_flow(
        self,
        ctx: Context,
        game: Game,
        status_json: Json,
        game_fp: str,
        recipient: Optional[Power] = None,
    ) -> bool:
        """
        1) if state has changed, re-compute wakeup time and re-generate message for review
        2) if message has not been reviewed, do nothing, return false
        3) if message has been rejected or does not exist, call generate_message_for_approval, return false
        4) if message has been approved, delete from cache and send message, return true
        """
        id_to_power = PRESS_COUNTRY_ID_TO_POWER

        state_changed = self.has_state_changed(ctx, game)
        message_review = get_message_review(game_fp)

        if status_json.get("pressType", "") == "RulebookPress" and (
            "R" in game.current_short_phase or "A" in game.current_short_phase
        ):
            if message_review:
                delete_message_review(game_fp)
                meta_annotations.after_message_generation_failed()
            return False

        if message_review and message_review["short_phase"] != game.current_short_phase:
            message_review["comment"] = (
                message_review["comment"]
                + " This message review was discarded due to phase mismatch."
            )
            set_message_review(game_fp, message_review)
            logging.info(
                f"Regenerating message review because its phase is out of date: {message_review['short_phase'], game.current_short_phase}"
            )
            meta_annotations.after_message_generation_failed()

            if self.has_phase_changed(ctx, game):
                logging.info("Detected phase change. Updating annotator.")
                meta_annotations.after_new_phase(game)

            self.generate_message_for_approval(game_fp, game, recipient=recipient)
            return False

        if state_changed:
            if message_review:
                message_review["comment"] = (
                    message_review["comment"]
                    + " This message review was discarded due to state change."
                )
                set_message_review(game_fp, message_review)
            logging.info("Regenerating message review due to state change.")
            meta_annotations.after_message_generation_failed()

            if self.has_phase_changed(ctx, game):
                logging.info("Detected phase change. Updating annotator.")
                meta_annotations.after_new_phase(game)

            self.generate_message_for_approval(game_fp, game, recipient=recipient)
            return False

        if not message_review:
            logging.info("Missing message review. Re-generating.")
            meta_annotations.after_message_generation_failed()
            self.generate_message_for_approval(game_fp, game, recipient=recipient)
            return False

        if not self.cfg.only_bump_msg_reviews_for_same_power:
            if (
                self.get_last_timestamp_or_zero(game_fp, game)
                != message_review["last_timestamp_when_produced"]
            ):
                logging.info(
                    """Aborting message submission for message proposal.
                    Message was not conditioned on most recent game state, likely due to webdip issues."""
                )
                message_review["comment"] = (
                    message_review["comment"]
                    + """. This message review was not conditioned on most recent game state, so it was never sent. "
                    Instead, a new message proposal was generated based on the most recent game state."""
                )
                meta_annotations.after_message_generation_failed()
                set_message_review(game_fp, message_review)
                self.generate_message_for_approval(game_fp, game, recipient=recipient)

                return False

        # Re-gen when proposal marked as stale
        if message_review["flag_as_stale"]:
            self.generate_message_for_approval(game_fp, game, recipient=recipient)
            return False

        logging.info(
            f"Procesing message review for game {ctx.gameID} and power {id_to_power[ctx.countryID]}"
        )

        if self.check_wakeup(ctx):
            logging.info("WOKE UP!")

        # this branch also handles cases where message review is off
        # (i.e. self.cfg.require_message_approval=False)
        if (
            self.check_wakeup(ctx)
            and (
                message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.APPROVED
                or (
                    not self.cfg.require_message_approval
                    and message_review["msg_proposals"][0]["target_power"]
                    != message_review["power"]
                )
            )
        ) or message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.FORCE_SEND:
            self.last_successful_message_time[ctx] = Timestamp.now()
            if message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.FORCE_SEND:
                logging.info(
                    "Message proposal being force sent. This should only be used for testing!"
                )
            else:
                logging.info("Message proposal approved.")

            msg: OutboundMessageDict = {
                "sender": message_review["power"],
                "recipient": message_review["msg_proposals"][0]["target_power"],
                "message": message_review["msg_proposals"][0]["msg"],
                "phase": game.current_short_phase,
            }
            logging.info("Trying to submit to Webdip.")
            status_json, timestamp = self.send_message(ctx, id_to_power, msg)
            logging.info(f"Message successfully sent at time {timestamp}")
            if recipient:
                assert msg["recipient"] == recipient, (recipient, msg)
            meta_annotations.after_message_add({**msg, "time_sent": timestamp})  # type: ignore
            game = webdip_state_to_game(status_json, stop_at_phase=self.cfg.check_phase)
            update_phase_message_history_state(
                game_fp=get_message_history_key(ctx),
                game=game,
                agent_power=PRESS_COUNTRY_ID_TO_POWER[ctx.countryID],
            )
            with pathlib.Path(game_fp).open("w") as stream:
                stream.write(game.to_json())
            meta_annotations.commit_annotations(game)
            flag_proposal_as_stale(game_fp)
            self.generate_message_for_approval(game_fp, game, recipient=recipient)

            return True
        elif message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.UNREVIEWED:
            logging.info("Message proposal not yet reviewed.")
            return False
        elif message_review["msg_proposals"][0]["approval_status"] == ApprovalStatus.REJECTED:
            self.last_successful_message_time[ctx] = Timestamp.now()
            logging.info("Message proposal rejected. Re-generating.")
            meta_annotations.after_message_generation_failed()
            self.generate_message_for_approval(game_fp, game, recipient=recipient)
            return False
        raise Exception(
            f"Unexpected behavior in run_message_approval_flow. This is message review: {json.dumps(message_review)}. Is bot awake? {self.check_wakeup(ctx)}"
        )

    @retry_on_connection_error
    def submit_orders(self, ctx: Context, game: Game, status_json: Json):
        variant_id = status_json["variantID"]
        id_to_power = COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP[variant_id]
        power = id_to_power[ctx.countryID]
        cfg = self.cfg
        try:
            agent_orders = self.players[ctx].get_orders(game)
        except Exception:
            tmp_path = os.path.abspath("game_exception.%s.json" % ctx.gameID)
            logging.error(
                f"Got exception while trying to get actions for {power}."
                f" Saving game to {tmp_path}"
            )
            game_json = safe_json_loads(game.to_json())
            with open(tmp_path, "w") as jf:
                json.dump(game_json, jf)
            raise

        logging.info(f"Power: {power} Phase: {game.current_short_phase} Orders: {agent_orders}")
        if cfg.present_timeout:
            try:
                logging.info(
                    f"Will sleep for {cfg.present_timeout} seconds before sending the orders."
                )
                logging.info("Press Ctrl-C to sent them now")
                logging.info("Press Ctrl-C twice to exit")
                time.sleep(cfg.present_timeout)
            except KeyboardInterrupt:
                print("Got Ctrl-C", flush=True)
                logging.warning("Got Ctrl-C. Will send orders in 5 secs unless see another Ctrl-C")
                time.sleep(5)

        if cfg.check_phase:
            return

        json_turn = status_json["phases"][-1]["turn"] if status_json["phases"] else 0
        if game.phase.startswith("W"):
            json_phase = "Builds"
        elif game.phase.endswith("RETREATS"):
            json_phase = "Retreats"
        else:
            json_phase = "Diplomacy"

        json_orders = [
            WebdipOrder(order, game=game, phase_type=game.phase_type, map_id=variant_id).to_dict()
            for order in agent_orders
        ]
        agent_orders_json = {
            "gameID": ctx.gameID,
            "countryID": ctx.countryID,
            "turn": json_turn,
            "phase": json_phase,
            "orders": json_orders,
            "ready": "Yes"
            if (
                self.cfg.ready_immediately
                or json_phase == "Retreats"
                or (status_json.get("pressType", "") == "RulebookPress" and json_phase == "Builds")
            )
            else "No",
        }

        coast_id_to_loc_id = _build_coast_id_to_loc_id(TERR_ID_TO_LOC_BY_MAP[variant_id])
        for order in agent_orders_json["orders"]:
            if order["fromTerrID"] in coast_id_to_loc_id:
                order["fromTerrID"] = coast_id_to_loc_id[order["fromTerrID"]]

        logging.info(f"JSON: {pformat(agent_orders_json)}")
        orders_resp = post_req(
            self.api_url, {"route": POST_ORDERS_ROUTE}, agent_orders_json, self.api_key
        )

        ###############################################################################
        # After this it's all sanity checks and corner cases
        ###############################################################################

        if check_orders_corner_cases(
            orders_resp, agent_orders_json, power, game, self.api_url, self.api_key, variant_id,
        ):
            return

        if orders_resp.status_code != 200:
            logging.error(f"Error {orders_resp.status_code}; Response: {orders_resp.content}")

            if orders_resp.status_code == 400 and any(
                orders_resp.content.startswith(x)
                for x in (b"Invalid turn, expected", b"Invalid phase")
            ):
                logging.warning(
                    "We probably sent orders for a phase that just changed: %s. Will carry on",
                    orders_resp.content,
                )
                return

        try:
            orders_resp_json = safe_json_loads(orders_resp.content)
        except WrappedJSONDecodeError as e:
            logger.info("GOT ERROR DECODING ORDER RESPONSE!")
            logger.info(orders_resp.content)
            logger.info(e)
            return

        logger.info(f"Response: {pformat(orders_resp_json)}")

        # sanity check that the orders were processed correctly
        order_req_by_unit = {
            x["terrID"] if x["terrID"] != "" else x["toTerrID"]: x
            for x in agent_orders_json["orders"]
        }
        order_resp_by_unit = {
            x["terrID"] if x["terrID"] is not None else x["toTerrID"]: x for x in orders_resp_json
        }
        if len(order_req_by_unit) > len(order_resp_by_unit):
            raise RuntimeError(f"{order_req_by_unit} != {order_resp_by_unit}")
        if len(order_req_by_unit) < len(order_resp_by_unit) and game.phase.endswith("MOVEMENT"):
            raise RuntimeError(f"{order_req_by_unit} != {order_resp_by_unit}")
        for terr in order_req_by_unit:
            if order_req_by_unit[terr]["type"] != order_resp_by_unit[terr]["type"]:
                if (
                    order_req_by_unit[terr]["type"] == "Destroy"
                    and order_resp_by_unit[terr]["type"] == "Retreat"
                ):
                    continue
                raise RuntimeError(f"{order_req_by_unit[terr]} != {order_resp_by_unit[terr]}")


def check_orders_corner_cases(
    orders_resp: Response,
    agent_orders_json: Dict[str, Any],
    power: Power,
    game: Game,
    api_url: str,
    api_key: str,
    variant_id: GameVariant,
):
    if (
        game.get_phase_history()
        and game.get_phase_history()[-1].orders.get(power)
        and (
            orders_resp.content.startswith(b"Invalid phase, expected `Retreats`, got ")
            and game.get_phase_history()[-1].name.endswith("R")
            or orders_resp.content.startswith(b"Invalid phase, expected `Builds`, got `Diplomacy`")
            and game.get_phase_history()[-1].name.endswith("A")
        )
    ):
        # In rare cases, webdip processes Retreat or Build orders and
        # updates the phase, but still requires us to re-submit those
        # orders a second time. We have not root- caused this behavior but
        # it's likely due to some race condition in webdip game processing.
        # Here we detect that case, reconstruct the orders from the
        # previous phase and submit them.
        prev_phase = "Retreats" if game.get_phase_history()[-1].name.endswith("R") else "Builds"
        logging.info(f"Detected buggy {prev_phase} ... re-issuing the same orders.")
        repeated_agent_orders_json = {k: v for k, v in agent_orders_json.items()}
        repeated_agent_orders_json.update(
            phase=prev_phase,
            orders=[
                WebdipOrder(
                    order, game=game, phase_type=game.phase_type, map_id=variant_id,
                ).to_dict()
                for order in game.get_phase_history()[-1].orders[power]
            ],
        )
        orders_resp = post_req(
            api_url, {"route": POST_ORDERS_ROUTE}, repeated_agent_orders_json, api_key
        )
        return True

    return False


def play_webdip(cfg: conf_cfgs.PlayWebdipTask):
    # swallow certain exceptions and log to slack instead
    GLOBAL_SLACK_EXCEPTION_SWALLOWER.activate()

    if heyhi.is_on_slurm():
        num_tasks = os.environ.get("SLURM_NTASKS", None)
        num_tasks = 1 if not num_tasks else int(num_tasks)
        if num_tasks > 1:
            # 6 workers for messages and 7th (when recipient = our power) for orders.
            assert num_tasks == len(
                POWERS
            ), f"Running un multiple tasks. Expected to get one task per power, but got {num_tasks}"
            assert (
                cfg.recipient is None
            ), "If running with auto-parallel-recipient, don't set recipient"
            global_rank = int(os.environ["SLURM_PROCID"])
            my_power = POWERS[global_rank]
            logging.info(f"Running in auto-parallel-recipient mode. My power is {my_power}")
            cfg_proto = cfg.to_editable()
            cfg_proto.recipient = my_power
            cfg = cfg_proto.to_frozen()

    assert cfg.agent is not None
    agent = build_agent_from_cfg(cfg.agent)

    if cfg.retry_exception_attempts is None:
        return _play_webdip_without_retries(cfg, agent)

    assert cfg.retry_exception_attempts is not None
    infinite_retries = cfg.retry_exception_attempts < 0
    left_attempts = cfg.retry_exception_attempts
    while True:
        gc.collect()
        last_attempt = time.monotonic()
        try:
            _play_webdip_without_retries(cfg, agent)
        except Exception as e:
            if time.monotonic() - last_attempt > RETRY_SUCCESS_TIME:
                # If _play_webdip_without_retries managed to survive long enough
                # before dying, we reset the counter.
                left_attempts = cfg.retry_exception_attempts
            if not infinite_retries and left_attempts <= 0:
                logging.exception("Got an exception. No attempts left. Going to raise")
                raise
            logging.exception(
                "Got an exception (%s). %s, so will try again in %s seconds",
                e,
                ("Has infinite retries" if infinite_retries else f"{left_attempts} attempts left"),
                RETRY_SLEEP_TIME,
            )
            time.sleep(RETRY_SLEEP_TIME)
            left_attempts -= 1
        else:
            # _play_webdip_without_retries may just exist, e.g., we download a single game.
            break


def _play_webdip_without_retries(cfg: conf_cfgs.PlayWebdipTask, agent: BaseAgent):
    logger = logging.getLogger()
    file_level = getattr(logger, "_file_level", logging.INFO)
    console_level = getattr(logger, "_console_level", logging.DEBUG)
    print(f"file_level= {file_level} console_level= {console_level}")
    log_dir = pathlib.Path(cfg.log_dir % dict(user=getpass.getuser()))
    if cfg.recipient:
        log_dir = log_dir / f"recipient_{cfg.recipient}"
    log_dir.mkdir(exist_ok=True, parents=True)
    game_dir = log_dir / "games"
    game_dir.mkdir(exist_ok=True, parents=True)

    api_url = cfg.webdip_url + API_PATH
    assert cfg.api_key
    assert cfg.account_name
    api_keys = cfg.api_key.split(",")
    account_names = cfg.account_name.split(",")
    assert len(api_keys) == len(account_names), (api_keys, account_names)

    def get_checkpoint_path_for_api_key(api_key: str) -> pathlib.Path:
        if cfg.checkpoint_path is None and cfg.game_id:
            assert False, """
            By default, agent state checkpoint_path is a per-api-key path in the log directory,
            and must be unique per simultaneous running process. If starting multiple processes
            per api_key to handle different game_ids, please specify checkpoint_path in the config
            appropriately as a distinct value per each process. You can also use %(user) and %(api_key)
            within the path.
            If default per-api-key behavior is still correct, specify checkpoint_path=PER_API_KEY.
            """

        if cfg.checkpoint_path is None or cfg.checkpoint_path == "PER_API_KEY":
            checkpoint_dir = log_dir / "checkpoints"
            checkpoint_path = get_default_checkpoint_path(
                checkpoint_dir, api_url=api_url, api_key=api_key
            )
        else:
            checkpoint_path = pathlib.Path(
                cfg.checkpoint_path % dict(user=getpass.getuser(), api_key=api_key)
            )
            assert (
                len(api_keys) <= 1
            ), "If specifying a fixed checkpoint path, please only run one process per api key"

        logging.info(f"Agent state checkpoint path for this api key is: {checkpoint_path}")
        checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
        return checkpoint_path

    heyhi.setup_logging(
        fpath=log_dir / "main.log", file_level=file_level, console_level=console_level,
    )
    logging.info("Will write logs to %s/{main.log,game_XXX_POWER.log}", log_dir)
    logging.info("Will save games to %s/game_XXX.json", game_dir)
    logging.info("Cfg:")
    logging.info(cfg)

    # Save CFG for testing in future
    with open(log_dir / "cfg", "w") as f:
        json.dump(cfg.to_dict(), f)

    game_ids = [int(x) for x in cfg.game_id.split(",")] if cfg.game_id else None
    game_name = cfg.game_name if cfg.game_name else None
    bots = [
        WebdipBotWrapper(
            cfg,
            agent,
            api_url,
            api_key,
            account_name,
            game_ids,
            game_name,
            game_dir,
            get_checkpoint_path_for_api_key(api_key),
            cfg.sleep_multiplier,
            expected_recipient=cfg.recipient,
        )
        for account_name, api_key in zip(account_names, api_keys)
    ]
    for bot in bots:
        bot.maybe_load_from_checkpoint(get_checkpoint_path_for_api_key(bot.api_key))
    last_bot = bots[0]
    last_ctx = None
    for bot in itertools.cycle(bots):
        if get_kill_switch():
            logging.info("Kill switch pulled! Sleeping for 5 seconds")
            time.sleep(5)
            continue
        if bot.cfg.is_backup and not get_should_run_backup(bot.account_name):
            logging.info("Backups are OFF and this bot IS a backup. Sleeping for 5 seconds.")
            time.sleep(5)
            continue
        if not bot.cfg.is_backup and get_should_run_backup(bot.account_name):
            logging.info("Backups are ON and this bot IS NOT a backup. Sleeping for 5 seconds.")
            time.sleep(5)
            continue

        last_bot = bot
        logging.info(f"========= Servicing API key {bot.api_key} =============")

        # START: Initialize logs, find game to process, and check if game can be processed
        heyhi.setup_logging(
            fpath=log_dir / "main.log", file_level=file_level, console_level=console_level,
        )

        # Keep-alive for restarting bots
        set_keep_alive_and_last_restart_ts(bot.account_name, keep_alive=Timestamp.now())

        #####
        ctx, missing_orders = bot.find_context()
        last_ctx = ctx
        logging.info(f"find_context returned: {ctx}")

        if ctx is None:
            if bot == bots[-1]:
                logger.info("Sleeping for 5 seconds.")
                time.sleep(5)
            continue

        try:
            status_json = bot.get_status_json(ctx)
        except WebdipGameNotFoundException:
            bot.purge_ctx(ctx, "Webdip game not found.")
            continue
        if status_json is None:
            continue

        if getattr(agent, "message_handler", None) is not None:
            game_pot_type = POT_TYPE_CONVERSION[status_json["potType"]]
            logging.info(f"Game pot type: {game_pot_type}")

        if status_json["variantID"] != cfg.variant_id and not cfg.json_out:
            logging.info(
                "Skipping game %s. It has non matching variant id %s. Expected %s",
                ctx.gameID,
                status_json["variantID"],
                cfg.variant_id,
            )
            continue

        if status_json["phases"]:
            logging.info(
                "XXXX %s .   %s %s %s %s %s",
                status_json["gameID"],
                status_json["turn"],
                status_json["phase"],
                status_json["phases"][-1]["turn"],
                status_json["phases"][-1]["phase"],
                len(status_json["phases"][-1]["orders"]),
            )

        id_to_power = COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP[status_json["variantID"]]
        power = id_to_power[ctx.countryID]

        bot_handles_dialogue = (
            cfg.allow_dialogue
            and bot.agent.can_sleep()
            and status_json.get("pressType", "") != "NoPress"
            and (cfg.recipient is None or cfg.recipient != power)
        )
        # Handling orders if we are NOT in parallel recipient mode OR we either
        # the order generation worker OR no-orders were submitted so far.
        bot_handles_orders = (
            cfg.recipient is None
            or cfg.recipient == power
            or "Saved" not in status_json["orderStatus"].split(",")
        )
        bot_handles_draws = bot_handles_orders

        game = webdip_state_to_game(status_json, stop_at_phase=cfg.check_phase)

        if log_dir is not None:
            heyhi.setup_logging(
                fpath=log_dir / f"game_{ctx.gameID}_{power}.log",
                file_level=file_level,
                console_level=console_level,
            )
        logging.info(
            "=" * 40 + " Servicing game: game=%s phase=%s power=%s recipient=%s",
            ctx.gameID,
            game.get_current_phase(),
            power,
            cfg.recipient,
        )
        logging.info(
            f"bot_handles_dialogue={bot_handles_dialogue} bot_handles_orders={bot_handles_orders} bot_handles_draws={bot_handles_draws}"
        )

        # Saving config for every phase to the log.
        # logging.info("Cfg:\n%s", cfg)
        game_fp = construct_game_fp(game_dir, ctx, id_to_power)
        with game_fp.open("w") as stream:
            stream.write(game.to_json())
        pathlib.Path(str(game_fp).replace(".json", "")).mkdir(exist_ok=True, parents=True)

        if cfg.json_out:
            game_json = safe_json_loads(game.to_json())
            with open(cfg.json_out, "w") as jf:
                json.dump(game_json, jf)
            return

        if agent is None:
            logging.info("No agent. Bailing after dumping game.")
            return

        if ctx not in bot.players:
            bot.players[ctx] = Player(bot.agent, power)
        # END: Initialize logs, find game to process, and check if game can be processed

        # START: Load or construct annotator
        annotations_out_name = pathlib.Path(str(game_fp).rsplit(".", 1)[0] + ".metann.jsonl")
        if ctx in bot.annotators:
            meta_annotations.push_annotator(bot.annotators[ctx])
        else:
            logging.info(f"No annotator found for {ctx}. Initializing one now...")
            meta_annotations.start_global_annotation(game, annotations_out_name)

        try:
            # Update annotator on state change by discarding existing annotations and updating phase-related metadata
            if bot.context_to_dialogue_state.get(ctx) and bot.context_to_dialogue_state.get(ctx)[0] != game.phase:  # type: ignore
                meta_annotations.after_message_generation_failed()
                meta_annotations.after_new_phase(game)
            # END: Load or construct annotator

            # START: Process press
            if bot_handles_dialogue:
                try:
                    sent_msg = bot.run_message_approval_flow(
                        ctx, game, status_json, str(game_fp), recipient=cfg.recipient
                    )
                except ShouldStopException:
                    logging.warning(
                        "Caught ShouldStopException - will start from the begining of the loop"
                    )
                    continue
                except WebdipSendMessageFailedException as e:
                    if (
                        hasattr(e, "message")
                        and e.message == RA_PHASE_MESSAGE_SEND_IN_RULEBOOK_ERROR_MESSAGE  # type: ignore
                    ):
                        logging.warning(
                            "Attempt to send message in R/A phase while in rulebook press was caught, and the message proposal will be deleted."
                        )
                        res = delete_message_review(str(game_fp))
                        logging.warning(f"Message proposal deletion: {res}")
                        sent_msg = False
                    else:
                        raise
                except (
                    socket.timeout,
                    urllib3.exceptions.ConnectTimeoutError,
                    urllib3.exceptions.MaxRetryError,
                    requests.exceptions.ConnectionError,
                    http.client.RemoteDisconnected,
                ) as e:
                    logging.exception("Uncaught network timeout, retrying without reloading agent")
                    continue

                if sent_msg:  # need to update game with new msg
                    status_json = bot.get_status_json(ctx)
                    assert status_json, ctx
                    game = webdip_state_to_game(status_json)
                    with game_fp.open("w") as stream:
                        stream.write(game.to_json())
            # END: Process press

            # START: Process orders
            if bot_handles_orders and bot.maybe_resend_orders(ctx, status_json):
                continue

            if bot_handles_draws and cfg.draw_on_stalemate_years:
                desired_draw_vote = (
                    game.get_consecutive_years_without_sc_change() >= cfg.draw_on_stalemate_years
                    and not game.any_sc_occupied_by_new_power()
                )
                new_status_json = set_draw_vote(ctx, desired_draw_vote)
                if new_status_json is None:
                    # On failure, go ahead and try to play orders anyways.
                    # The draw vote is in a broken/unknown state but do the best we can
                    logger.error("FAILED to set_draw_vote, attempting to continue anyways")
                else:
                    status_json = new_status_json

            if bot_handles_orders:
                cur_dialogue_state = (game.phase, bot.get_message_history_length(ctx, game))
                last_dialogue_state = bot.context_to_dialogue_state.get(ctx)
                if cur_dialogue_state != last_dialogue_state or missing_orders:
                    logger.info(
                        f"Orderable locations: {game.get_orderable_locations().get(power)}"
                    )

                    if (
                        status_json.get("pressType", "") == "RulebookPress"
                        and game.get_metadata("phase_minutes") == "5"
                        and ("R" in game.current_short_phase or "A" in game.current_short_phase)
                    ):
                        sec_to_sleep_before_send_orders = random.sample(range(5, 15), 1)[
                            0
                        ]  # sleep between 5 and 30 seconds
                        logger.info(
                            f"Due to being in rulebook press, 5min phases, and in R/A phase, will sleep {sec_to_sleep_before_send_orders} (~unif([5,15])) seconds before sending orders"
                        )
                        time.sleep(sec_to_sleep_before_send_orders)
                        logger.info(f"Woke up! Attempting to send orders now.")

                    bot.submit_orders(ctx, game, status_json)
                    if cfg.check_phase or cfg.force:
                        bot.post_process(ctx)
                        return

            # END: Process orders

            bot.context_to_dialogue_state[ctx] = (
                game.phase,
                bot.get_message_history_length(ctx, game),
            )
        finally:
            if last_ctx:
                last_bot.post_process(last_ctx)  # In case process terminated at weird point
