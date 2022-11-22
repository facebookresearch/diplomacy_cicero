#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import Counter, defaultdict
import collections
from datetime import datetime, timezone
import functools
import glob
import itertools
import pathlib
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict
import redis
import re
from fairdiplomacy import pydipcc
from fairdiplomacy.data.build_dataset import COUNTRY_ID_TO_POWER_OR_ALL
from conf import conf_cfgs
from fairdiplomacy.game import POWERS, sort_phase_key_string
from fairdiplomacy.timestamp import Timestamp
import json
import uuid
from pprint import pprint, pformat
import logging

from fairdiplomacy.typedefs import Power, StrEnum, Context
from fairdiplomacy.viz.meta_annotations.annotator import MetaAnnotator
from parlai_diplomacy.wrappers.classifiers import INF_SLEEP_TIME

"""
FvA games will not have press, so we can assume that variant type is classic.

NOTE: Some hard-coded global constants need to be set for this work (REDIS_IP, PORT, WEBDIP_GAMES_BASE_DIR).

"""
PRESS_COUNTRY_ID_TO_POWER = COUNTRY_ID_TO_POWER_OR_ALL
PRESS_POWER_TO_COUNTRY_ID = {v: k for k, v in PRESS_COUNTRY_ID_TO_POWER.items()}

REDIS_IP = None
# Production
PROD_DB = 1
DEV_DB = 2
PORT = None

WEBDIP_GAMES_BASE_DIR = None
WEBDIP_GAMES_DIR_GLOB = f"{WEBDIP_GAMES_BASE_DIR}/**/games"

if REDIS_IP is None or PORT is None or WEBDIP_GAMES_BASE_DIR is None:
    raise NotImplementedError("This global variables must be set.")

MESSAGE_REVIEW_CODEBASE_VERSION = 1


# A string represents concatenation of phase and message history. Used to check
# if game changed. The lexiographic ordering on hashes correspods to the
# ordering within a game.
PhaseMessageHistoryHash = str

PHASE_MESSAGE_HISTORY_HASH_SUFFIX = ":message_history_state"
PHASE_MESSAGE_HISTORY_TTL_SECONDS = 60 * 5  # 5 minutes.


class MessageApprovalRedisCacheException(Exception):
    pass


@functools.lru_cache(None)
def get_redis_host(db: Optional[int] = None) -> redis.Redis:
    try:
        redis_host = redis.Redis(
            host=REDIS_IP, port=PORT, db=db if db else PROD_DB, decode_responses=True
        )

        redis_cache_version = redis_host.get("message_review_version")
        assert (
            redis_cache_version and int(redis_cache_version) == MESSAGE_REVIEW_CODEBASE_VERSION
        ), f"Version of the database ({redis_cache_version}) does not match version of the codebase ({MESSAGE_REVIEW_CODEBASE_VERSION})"
    except Exception as e:
        raise MessageApprovalRedisCacheException(f"get_redis_host failed: {e}")

    return redis_host


class ApprovalStatus(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNREVIEWED = "UNREVIEWED"
    FORCE_SEND = "FORCE_SEND"


"""
This types the schema used for values in the message review Redis cache.
"""


class MessageProposal(TypedDict):
    target_power: str
    msg: str
    approval_status: ApprovalStatus
    tags: List[str]


class MessageReviewData(TypedDict):
    id: int
    game_id: int
    power: str
    msg_proposals: List[MessageProposal]
    wakeup_time: Timestamp
    last_timestamp_when_produced: Timestamp
    last_serviced: Timestamp
    user: str
    comment: str
    annotator: Optional[MetaAnnotator]
    cfg: Optional[str]
    phase_end_timestamp: Optional[Timestamp]
    short_phase: Optional[str]
    parallel_recipient: Optional[str]
    flag_as_stale: bool


def gen_id() -> int:
    return uuid.uuid4().int


def botgame_fp_to_context(game_fp: str) -> Context:
    parsed_game_fp = re.search(r"game_(\d+)_(.+).json", game_fp)
    assert parsed_game_fp and len(parsed_game_fp.groups()) == 2
    gameID = int(parsed_game_fp.groups()[0])
    power = parsed_game_fp.groups()[1]

    return Context(gameID, PRESS_POWER_TO_COUNTRY_ID[power])


def get_message_review(game_fp: str, db: Optional[int] = None) -> Optional[MessageReviewData]:
    """
    All calls to get message reviews in Redis database should be hidden behind this method.
    """
    redis_host = get_redis_host(db)
    try:
        message_proposal_metadata = redis_host.hgetall(game_fp)

        if not message_proposal_metadata:
            return None

        num_proposals = int(message_proposal_metadata["num_proposals"])

        message_proposals = []
        for i in range(num_proposals):
            message_proposals.append(redis_host.hgetall(game_fp + f":{i}"))

        annotator = None
        if message_proposal_metadata.get("annotator", "") != "":
            if len(game_fp.split(":")) > 1:
                game_fp = game_fp.split(":")[1]
            with open(game_fp, "r") as f:
                game = pydipcc.Game.from_json(f.read())
            annotations_out_name = pathlib.Path(
                str(game_fp).rsplit(".", 1)[0] + "_REDIS.metann.jsonl"
            )
            annotator = MetaAnnotator(game, annotations_out_name, silent=True)
            annotator.load_state_dict(json.loads(message_proposal_metadata["annotator"]))

        res: MessageReviewData = {
            "id": int(message_proposal_metadata["id"]),
            "power": message_proposal_metadata["power"],
            "game_id": int(message_proposal_metadata["game_id"])
            if "game_id" in message_proposal_metadata
            else -1,
            "msg_proposals": [
                {
                    "target_power": msg_proposal["target_power"],
                    "msg": msg_proposal["msg"],
                    "approval_status": ApprovalStatus(msg_proposal["approval_status"]),
                    "tags": msg_proposal["tags"].split(",") if msg_proposal["tags"] != "" else [],
                }
                for msg_proposal in message_proposals
            ],
            "wakeup_time": Timestamp.from_centis(message_proposal_metadata["wakeup_time"]),
            "last_timestamp_when_produced": Timestamp.from_centis(
                message_proposal_metadata["last_timestamp_when_produced"]
            ),
            "last_serviced": Timestamp.from_centis(message_proposal_metadata["last_serviced"]),
            "user": message_proposal_metadata["user"],
            "comment": message_proposal_metadata["comment"],
            "annotator": annotator,
            "parallel_recipient": message_proposal_metadata.get("parallel_recipient", ""),
            "cfg": message_proposal_metadata.get("cfg", "")
            if message_proposal_metadata.get("cfg", "") != ""
            else None,
            "phase_end_timestamp": Timestamp.from_centis(
                message_proposal_metadata["phase_end_timestamp"]
            )
            if "phase_end_timestamp" in message_proposal_metadata
            and message_proposal_metadata["phase_end_timestamp"] != ""
            else None,
            "short_phase": message_proposal_metadata.get("short_phase", "")
            if message_proposal_metadata.get("short_phase", "") != ""
            else None,
            "flag_as_stale": bool(int(message_proposal_metadata.get("flag_as_stale", "0"))),
        }
        return res
    except (redis.exceptions.RedisClusterException, KeyError) as e:
        raise MessageApprovalRedisCacheException(f"get_message_review failed: {e}")


def set_message_review(
    game_fp: str,
    message_review_data: MessageReviewData,
    db: Optional[int] = None,
    archive: bool = True,
    review_update_only: bool = False,
) -> str:
    """
    All calls to create or update message reviews in Redis database should be hidden behind this method.
    An archive is created upon message review creation, and updated whenever a field is changed. The key
    that identifies identical message reviews is the "id" field. There is only ever at most one archive for each
    id.

    Since Redis does not allow for nested hashtables, I store the List[MessageProposal] field "msg_proposals"
    as additional hash entries that have an appended index in their key name. Then, a field "num_proposals" in the top-level
    hash table provides guidance for how many additional hash entries there are, enabling the getter method to reconstruct
    the "msg_proposals" field.
    """
    redis_host = get_redis_host(db)
    try:
        cur_review = get_message_review(game_fp, db)
        if cur_review and review_update_only and cur_review["id"] != message_review_data["id"]:
            raise MessageApprovalRedisCacheException(
                f"Trying to update unexpected proposal! Current cache message is ({cur_review['msg_proposals'][0]['msg']}) but new message is ({message_review_data['msg_proposals'][0]['msg']})"
            )

        redis_host.hmset(
            game_fp,
            {
                "id": str(message_review_data["id"]),
                "power": message_review_data["power"],
                "wakeup_time": message_review_data["wakeup_time"].to_centis(),
                "last_timestamp_when_produced": message_review_data[
                    "last_timestamp_when_produced"
                ].to_centis(),
                "last_serviced": message_review_data["last_serviced"].to_centis(),
                "user": message_review_data["user"],
                "comment": message_review_data["comment"],
                "num_proposals": len(message_review_data["msg_proposals"]),
                "annotator": (
                    json.dumps(message_review_data["annotator"].to_dict())
                    if message_review_data["annotator"]
                    else ""
                ),
                "cfg": message_review_data["cfg"] if message_review_data["cfg"] else "",
                "phase_end_timestamp": message_review_data["phase_end_timestamp"].to_centis()
                if message_review_data["phase_end_timestamp"]
                else "",
                "short_phase": message_review_data["short_phase"]
                if message_review_data["short_phase"] is not None
                else "",
                "parallel_recipient": message_review_data["parallel_recipient"] or "",
                "game_id": message_review_data["game_id"] or "",
                "flag_as_stale": str(int(message_review_data["flag_as_stale"])),
            },
        )

        for i, msg_proposal in enumerate(message_review_data["msg_proposals"]):
            redis_host.hmset(
                game_fp + f":{i}",
                {
                    "target_power": msg_proposal["target_power"],
                    "msg": msg_proposal["msg"],
                    "approval_status": msg_proposal["approval_status"].name,
                    "tags": ",".join(msg_proposal["tags"]),
                },
            )
    except (redis.exceptions.RedisClusterException, MessageApprovalRedisCacheException) as e:
        raise MessageApprovalRedisCacheException(f"set_message_review failed: {e}")

    if archive:
        archive_message_review(game_fp, db)

    return f"Succesfully set message review cache entry for {game_fp}"


def delete_message_review(game_fp: str, db: Optional[int] = None, archive: bool = True) -> str:
    """
    This method will remove message review from cache and update its archive.
    """
    redis_host = get_redis_host(db)
    try:
        if not get_message_review(game_fp, db):
            return f"No message review found to delete!"

        if archive:
            archive_message_review(game_fp, db)

        message_proposal_metadata = redis_host.hgetall(game_fp)
        num_proposals = int(message_proposal_metadata["num_proposals"])

        redis_host.delete(game_fp)
        for i in range(num_proposals):
            redis_host.delete(game_fp + f":{i}")
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"delete_message_review failed: {e}")

    return f"Successfully deleted message review cache entry for {game_fp}"


def delete_archive(game_fp: str, db: Optional[int] = None):
    """
    This method will delete the archive for a message review. This is used in tests.
    """
    redis_host = get_redis_host(db)
    try:
        ids = redis_host.smembers("archive:" + game_fp)

        for id in ids:
            delete_message_review("archive:" + game_fp + f":{id}", db, archive=False)

        redis_host.delete("archive:" + game_fp)
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"delete_message_review failed: {e}")

    return f"Successfully delete archived message reviews for {game_fp}"


def archive_message_review(game_fp: str, db: Optional[int] = None) -> str:
    """
    This method creates an archive for a message review. Each archive is keyed by the message review's
    name (e.g. game filepath) and its id field. This means that you can update fields in a message review, archive
    over and over, and still maintain only a single archive.

    Under the hood, this method uses a set to ensure that there is never more than one archive per message review
    (again, identified by its id).
    """
    redis_host = get_redis_host(db)
    try:
        res = get_message_review(game_fp, db)
        assert res
        id = str(res["id"])
        redis_host.sadd("archive:" + game_fp, id)

        set_message_review("archive:" + game_fp + f":{id}", res, db, archive=False)
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"delete_message_review failed: {e}")

    return f"Successfully archived message review cache entry for {game_fp}"


def get_archived_message_reviews(
    game_fp: str, db: Optional[int] = None
) -> List[MessageReviewData]:
    """
    Getter method to grab archived message reviews.
    """
    redis_host = get_redis_host(db)
    try:
        ids = redis_host.smembers("archive:" + game_fp)
        reviews = []
        for id in ids:
            res = get_message_review("archive:" + game_fp + f":{id}", db)
            if res:
                reviews.append(res)

    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"delete_message_review failed: {e}")

    return reviews


def get_kill_switch(db: Optional[int] = None) -> bool:
    """
    Flag to disable bots
    """
    redis_host = get_redis_host(db)
    try:
        flag = redis_host.get(f"WEBDIP_KILL_SWITCH?")
        return flag is not None and flag == "true"
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"get_test_games failed: {e}")


def get_should_run_backup(account_name: str, db: Optional[int] = None) -> bool:
    """
    Flag to turn on backup bots
    """
    redis_host = get_redis_host(db)
    try:
        flag = redis_host.get(f"{account_name}_SHOULD_USE_BACKUP?")
        return flag is not None and flag == "true"
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"get_test_games failed: {e}")


def get_test_games(db: Optional[int] = None) -> List[int]:
    """
    Getter method to see what games are test games. We only ever have 1 or 2 test games running, and they already
    require manual intervention to set up. Therefore, I manually update this field in the cache when I create/delete a test game.
    """
    redis_host = get_redis_host(db)
    try:
        game_ids = redis_host.smembers("current_test_games")
        return [int(game_id) for game_id in list(game_ids)]
    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"get_test_games failed: {e}")


def game_dir_to_message_reviews(
    game_dir: str, db: Optional[int] = None, key_on_context: bool = False
) -> Dict[str, MessageReviewData]:
    """
    Getter method to pull all message reviews for all games from a directory containing multiple game jsons.
    """

    game_fps = glob.glob(f"{game_dir}/game_*_*.json")
    games = {}
    for game_fp in game_fps:
        try:
            ctx = botgame_fp_to_context(game_fp)
            power = PRESS_COUNTRY_ID_TO_POWER[ctx.countryID]

            res = get_message_review(game_fp, db)
            if not res:
                continue

            if key_on_context:
                games[f"{ctx.gameID},{power}"] = res
            else:
                games[game_fp] = res
        except (redis.exceptions.RedisClusterException, KeyError) as e:
            raise MessageApprovalRedisCacheException(f"game_dir_to_message_reviews failed: {e}")

    return games


def game_dir_to_archived_message_reviews(
    game_dir: str, db: Optional[int] = None, key_on_context: bool = False
) -> Dict[str, MessageReviewData]:
    """
    Getter method to pull all message reviews for all games from a directory containing multiple game jsons.
    """

    game_fps = glob.glob(f"{game_dir}/**/game_*_*.json", recursive=True)
    games = {}
    for game_fp in game_fps:
        games[game_fp] = []
        try:
            ctx = botgame_fp_to_context(game_fp)

            revs = get_archived_message_reviews(game_fp, db)
            if len(revs) == 0:
                continue

            games[game_fp] += revs
        except (redis.exceptions.RedisClusterException, KeyError) as e:
            raise MessageApprovalRedisCacheException(f"game_dir_to_message_reviews failed: {e}")

    return games


def filter_hidden_games(
    revs: Dict[str, MessageReviewData], games_to_hide: Optional[List[int]] = None,
) -> Dict[str, MessageReviewData]:
    if games_to_hide is None:
        games_to_hide = get_test_games()
    return {
        game_fp: rev
        for game_fp, rev in revs.items()
        if botgame_fp_to_context(game_fp).gameID not in games_to_hide
    }


def get_webdip_games_archived_message_reviews() -> Dict[str, List[MessageReviewData]]:
    game_dirs = glob.glob(WEBDIP_GAMES_DIR_GLOB)

    game_fp_to_message_revs = {}
    for game_dir in game_dirs:
        game_fp_to_message_revs = {
            **game_fp_to_message_revs,
            **filter_hidden_games(game_dir_to_archived_message_reviews(game_dir)),
        }

    return game_fp_to_message_revs


def get_webdip_games_archived_message_review_summary() -> Dict:
    game_fp_to_message_revs = get_webdip_games_archived_message_reviews()

    def safe_divide(numerator, denominator) -> Optional[float]:
        if denominator == 0:
            return None
        else:
            return numerator / denominator

    summary = {}
    for game_fp, message_revs in game_fp_to_message_revs.items():
        ctx = botgame_fp_to_context(game_fp)
        num_reviews_by_approval_status = {
            approval_status.value: len(
                [
                    rev
                    for rev in message_revs
                    if rev["msg_proposals"][0]["approval_status"] == approval_status
                ]
            )
            for approval_status in ApprovalStatus
        }

        inf_sleep_force_sends = [
            rev
            for rev in message_revs
            if (rev["msg_proposals"][0]["approval_status"] == ApprovalStatus.FORCE_SEND)
            and (
                rev["wakeup_time"]
                > (
                    Timestamp.from_seconds(datetime.now(timezone.utc).timestamp())
                    + min(INF_SLEEP_TIME, Timestamp.from_seconds(1e9))
                    - Timestamp.from_seconds(5 * 365 * 24 * 60 * 60)  # 5 year buffer
                )
            )
        ]
        inf_sleep_force_sends_rate = safe_divide(
            len(inf_sleep_force_sends), num_reviews_by_approval_status[ApprovalStatus.FORCE_SEND]
        )

        rejection_rate_without_force_send = safe_divide(
            num_reviews_by_approval_status[ApprovalStatus.REJECTED],
            (
                num_reviews_by_approval_status[ApprovalStatus.APPROVED]
                + num_reviews_by_approval_status[ApprovalStatus.REJECTED]
            ),
        )

        rejection_rate_with_force_send = safe_divide(
            num_reviews_by_approval_status[ApprovalStatus.REJECTED],
            (
                num_reviews_by_approval_status[ApprovalStatus.APPROVED]
                + num_reviews_by_approval_status[ApprovalStatus.REJECTED]
                + num_reviews_by_approval_status[ApprovalStatus.FORCE_SEND]
            ),
        )

        force_sends_percent_of_sent_messages = safe_divide(
            num_reviews_by_approval_status[ApprovalStatus.FORCE_SEND],
            (
                num_reviews_by_approval_status[ApprovalStatus.FORCE_SEND]
                + num_reviews_by_approval_status[ApprovalStatus.APPROVED]
            ),
        )

        reviews_grouped_by_phase = {}
        with open(game_fp, "r") as f:
            orig_game = pydipcc.Game.from_json(f.read())
        for rev in message_revs:
            game = orig_game.rolled_back_to_timestamp_end(rev["last_timestamp_when_produced"])
            approval_status = rev["msg_proposals"][0]["approval_status"]
            current_phase = game.current_short_phase

            if current_phase not in reviews_grouped_by_phase:
                reviews_grouped_by_phase[current_phase] = {}

            if approval_status not in reviews_grouped_by_phase[current_phase]:
                reviews_grouped_by_phase[current_phase][approval_status] = []

            reviews_grouped_by_phase[current_phase][approval_status].append(rev)

        rejection_rate_without_force_send_by_phase = {}
        for phase in reviews_grouped_by_phase.keys():
            num_rejections_in_phase = len(
                reviews_grouped_by_phase[phase].get(ApprovalStatus.REJECTED, [])
            )
            num_approvals_in_phase = len(
                reviews_grouped_by_phase[phase].get(ApprovalStatus.APPROVED, [])
            )

            phase_rejection_rate_without_force_send = safe_divide(
                num_rejections_in_phase, num_rejections_in_phase + num_approvals_in_phase
            )

            rejection_rate_without_force_send_by_phase[phase] = (
                phase_rejection_rate_without_force_send,
                {"rejects": num_rejections_in_phase, "approvals": num_approvals_in_phase},
            )

        summary[ctx] = {
            "game_fp": game_fp,
            "logs_fp": game_fp.replace("/games/", "/").replace(".json", ".log"),
            "num_proposals_generated": sum(num_reviews_by_approval_status.values()),
            "num_proposals_reviewed": sum(
                [
                    num_reviews
                    for status, num_reviews in num_reviews_by_approval_status.items()
                    if status != ApprovalStatus.UNREVIEWED
                ]
            ),
            "num_reviews_by_approval_status": num_reviews_by_approval_status,
            "rejection_rate_without_force_send": rejection_rate_without_force_send,
            "rejection_rate_with_force_send": rejection_rate_with_force_send,
            "rejection_rate_without_force_send_by_phase": rejection_rate_without_force_send_by_phase,
            "inf_sleep_percent_of_force_sends": inf_sleep_force_sends_rate,
            "force_sends_percent_of_sent_messages": force_sends_percent_of_sent_messages,
            "tags": Counter(
                [
                    tag
                    for rev in message_revs
                    for proposal in rev["msg_proposals"]
                    for tag in proposal["tags"]
                ]
            ),
        }

    return summary


def maybe_load_agent_config_as_proto(agent_config_json: str) -> Optional[conf_cfgs.PlayWebdipTask]:
    if not agent_config_json:
        return None
    agent_config_dict = json.loads(agent_config_json)
    # Migrating old names.
    if "forced_recipient" in agent_config_dict:
        agent_config_dict["recipient"] = agent_config_dict.pop("forced_recipient")
    try:
        return conf_cfgs.PlayWebdipTask(**agent_config_dict)
    except ValueError as e:
        logging.error(
            "Failed to parse agent config in the message approval cache.\nErr: %s\nCfg:\n%s",
            e,
            pformat(agent_config_dict, indent=2),
        )
        return None


def compute_phase_message_history_state_with_power(
    game: pydipcc.Game, agent_power: Power
) -> Dict[Power, PhaseMessageHistoryHash]:
    """A dict that maps a power name to a game-dialogue state with this power.

    For all powers except agent_power the value will only be affected by the messages with that power.

    For agent_power we will account for all messages including message to ALL.
    """
    recipient_sender_pairs = [
        (msg_dct["sender"], msg_dct["recipient"])
        for phase_data in game.get_all_phases()
        for msg_dct in phase_data.messages.values()
    ]
    dialoge_with_agent = itertools.chain.from_iterable(
        (sender, recipient)
        for (sender, recipient) in recipient_sender_pairs
        if agent_power in (sender, recipient) and "ALL" not in (sender, recipient)
    )

    message_history_lengths = collections.Counter(dialoge_with_agent)
    message_history_lengths[agent_power] = len(recipient_sender_pairs)
    phase = sort_phase_key_string(game.current_short_phase)
    return {power: "%s_%010d" % (phase, message_history_lengths[power]) for power in POWERS}


def update_phase_message_history_state(
    game_fp: str, game: pydipcc.Game, agent_power: Power, db: Optional[int] = None
) -> None:
    redis_client = get_redis_host(db=db)
    key = f"{game_fp}{PHASE_MESSAGE_HISTORY_HASH_SUFFIX}"
    redis_client.hmset(
        key, compute_phase_message_history_state_with_power(game, agent_power=agent_power),  # type: ignore
    )
    redis_client.expire(key, PHASE_MESSAGE_HISTORY_TTL_SECONDS)


def maybe_get_phase_message_history_state(
    game_fp: str, db: Optional[int] = None
) -> Optional[Dict[Power, PhaseMessageHistoryHash]]:
    redis_client = get_redis_host(db=db)
    key = f"{game_fp}{PHASE_MESSAGE_HISTORY_HASH_SUFFIX}"
    return redis_client.hgetall(key) if redis_client.exists(key) else None


def flag_proposal_as_stale(game_fp: str, db: Optional[int] = None) -> str:
    """
    This method will mark a message review as stale
    """
    try:
        rev = get_message_review(game_fp, db)
        if not rev:
            return f"No message review found to delete!"

        rev["flag_as_stale"] = True
        set_message_review(game_fp, rev, db)

    except redis.exceptions.RedisClusterException as e:
        raise MessageApprovalRedisCacheException(f"flag_proposal_as_stale failed: {e}")

    return f"Successfully marked as stale: {game_fp}"


def get_parallel_recipient_archived_message_reviews(
    log_dir: str, backup_log_dir: Optional[str] = None
) -> Dict[str, List[Tuple[str, MessageReviewData]]]:
    per_recipient_game_dirs = glob.glob(log_dir + "/**/games", recursive=True)
    per_recipient_game_fp_to_message_revs = {}
    for game_dir in per_recipient_game_dirs:
        per_recipient_game_fp_to_message_revs = {
            **per_recipient_game_fp_to_message_revs,
            **filter_hidden_games(game_dir_to_archived_message_reviews(game_dir)),
        }

    if backup_log_dir:
        backup_per_recipient_game_dirs = glob.glob(backup_log_dir + "/**/games", recursive=True)
        for game_dir in backup_per_recipient_game_dirs:
            per_recipient_game_fp_to_message_revs = {
                **per_recipient_game_fp_to_message_revs,
                **filter_hidden_games(game_dir_to_archived_message_reviews(game_dir)),
            }

    game_stem_to_message_revs = {}
    for per_recipient_game_fp, revs in per_recipient_game_fp_to_message_revs.items():
        stem = str(pathlib.Path(per_recipient_game_fp).stem)
        if stem not in game_stem_to_message_revs:
            game_stem_to_message_revs[stem] = []

        for rev in revs:
            game_stem_to_message_revs[stem].append((per_recipient_game_fp, rev))

    return game_stem_to_message_revs


def get_parallel_recipient_archived_message_review_stats(
    log_dir: str, backup_log_dir: Optional[str] = None
) -> Dict[str, List[MessageReviewData]]:
    per_recipient_game_dirs = glob.glob(log_dir + "/**/games", recursive=True)
    per_recipient_game_fp_to_message_revs = {}
    for game_dir in per_recipient_game_dirs:
        per_recipient_game_fp_to_message_revs = {
            **per_recipient_game_fp_to_message_revs,
            **filter_hidden_games(game_dir_to_archived_message_reviews(game_dir)),
        }

    if backup_log_dir:
        backup_per_recipient_game_dirs = glob.glob(backup_log_dir + "/**/games", recursive=True)
        for game_dir in backup_per_recipient_game_dirs:
            per_recipient_game_fp_to_message_revs = {
                **per_recipient_game_fp_to_message_revs,
                **filter_hidden_games(game_dir_to_archived_message_reviews(game_dir)),
            }

    game_stem_to_message_revs = {}
    for per_recipient_game_fp, revs in per_recipient_game_fp_to_message_revs.items():
        stem = str(pathlib.Path(per_recipient_game_fp).stem)
        if stem not in game_stem_to_message_revs:
            game_stem_to_message_revs[stem] = []

        for rev in revs:
            game_stem_to_message_revs[stem].append(rev)

    unique_game_stems = game_stem_to_message_revs.keys()

    data = {
        "unique_game_stems": unique_game_stems,
        "review_stats": {
            game_stem: Counter([rev["msg_proposals"][0]["approval_status"] for rev in revs])
            for game_stem, revs in game_stem_to_message_revs.items()
        },
        "rejected_msgs": {
            game_stem: [
                rev
                for rev in revs
                if rev["msg_proposals"][0]["approval_status"] == ApprovalStatus.REJECTED
            ]
            for game_stem, revs in game_stem_to_message_revs.items()
        },
    }

    data["rejection_rates"] = {}
    data["total_msg_counts"] = {}
    for game_stem, review_stats in data["review_stats"].items():
        if (review_stats[ApprovalStatus.APPROVED] + review_stats[ApprovalStatus.REJECTED]) > 0:
            rejection_rate = review_stats[ApprovalStatus.REJECTED] / (
                review_stats[ApprovalStatus.APPROVED] + review_stats[ApprovalStatus.REJECTED]
            )
        else:
            rejection_rate = None
        data["rejection_rates"][game_stem] = rejection_rate
        data["total_msg_counts"][game_stem] = (
            review_stats[ApprovalStatus.APPROVED] + review_stats[ApprovalStatus.REJECTED]
        )

    return data


if __name__ == "__main__":
    pprint(get_webdip_games_archived_message_review_summary())
