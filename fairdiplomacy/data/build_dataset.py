#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import enum
import joblib
import json
import logging
import os
import sqlite3
import traceback
from collections import defaultdict
from glob import glob
from pprint import pformat
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from fairdiplomacy.pydipcc import Game
from fairdiplomacy.typedefs import Power, Timestamp, Phase
from fairdiplomacy.webdip.utils import turn_to_phase


class GameVariant(enum.IntFlag):
    CLASSIC = 1
    FVA = 15


TABLE_GAMES = "redacted_games"
TABLE_MOVES = "redacted_movesarchive"
TABLE_MESSAGES = "redacted_messages"

# Created by running:
#
# TERR_ID_TO_LOC = {
#     terr_id: FULL_TO_SHORT[full_name]
#     for (terr_id, full_name) in db.execute("SELECT id, name FROM wD_Territories WHERE mapID=1")
# }
TERR_ID_TO_LOC = {
    0: "",
    1: "CLY",
    2: "EDI",
    3: "LVP",
    4: "YOR",
    5: "WAL",
    6: "LON",
    7: "POR",
    8: "SPA",
    9: "NAF",
    10: "TUN",
    11: "NAP",
    12: "ROM",
    13: "TUS",
    14: "PIE",
    15: "VEN",
    16: "APU",
    17: "GRE",
    18: "ALB",
    19: "SER",
    20: "BUL",
    21: "RUM",
    22: "CON",
    23: "SMY",
    24: "ANK",
    25: "ARM",
    26: "SYR",
    27: "SEV",
    28: "UKR",
    29: "WAR",
    30: "LVN",
    31: "MOS",
    32: "STP",
    33: "FIN",
    34: "SWE",
    35: "NWY",
    36: "DEN",
    37: "KIE",
    38: "BER",
    39: "PRU",
    40: "SIL",
    41: "MUN",
    42: "RUH",
    43: "HOL",
    44: "BEL",
    45: "PIC",
    46: "BRE",
    47: "PAR",
    48: "BUR",
    49: "MAR",
    50: "GAS",
    51: "BAR",
    52: "NWG",
    53: "NTH",
    54: "SKA",
    55: "HEL",
    56: "BAL",
    57: "BOT",
    58: "NAO",
    59: "IRI",
    60: "ENG",
    61: "MAO",
    62: "WES",
    63: "LYO",
    64: "TYS",
    65: "ION",
    66: "ADR",
    67: "AEG",
    68: "EAS",
    69: "BLA",
    70: "TYR",
    71: "BOH",
    72: "VIE",
    73: "TRI",
    74: "BUD",
    75: "GAL",
    76: "SPA/NC",
    77: "SPA/SC",
    78: "STP/NC",
    79: "STP/SC",
    80: "BUL/EC",
    81: "BUL/SC",
}

COUNTRY_ID_TO_POWER = {
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}

COUNTRY_ID_TO_POWER_OR_ALL = {
    0: "ALL",
    1: "ENGLAND",
    2: "FRANCE",
    3: "ITALY",
    4: "GERMANY",
    5: "AUSTRIA",
    6: "TURKEY",
    7: "RUSSIA",
}

# Built with bin/compute_names_fvsa.py.
TERR_ID_TO_LOC_FRANCE_VS_AUSTRIA = {
    0: "",
    1: "CLY",
    2: "EDI",
    3: "LVP",
    4: "YOR",
    5: "WAL",
    6: "LON",
    7: "POR",
    8: "SPA",
    9: "SPA/NC",
    10: "SPA/SC",
    11: "NAF",
    12: "TUN",
    13: "NAP",
    14: "ROM",
    15: "TUS",
    16: "PIE",
    17: "VEN",
    18: "APU",
    19: "GRE",
    20: "ALB",
    21: "SER",
    22: "BUL",
    23: "BUL/EC",
    24: "BUL/SC",
    25: "RUM",
    26: "CON",
    27: "SMY",
    28: "ANK",
    29: "ARM",
    30: "SYR",
    31: "SEV",
    32: "UKR",
    33: "WAR",
    34: "LVN",
    35: "MOS",
    36: "STP",
    37: "STP/NC",
    38: "STP/SC",
    39: "FIN",
    40: "SWE",
    41: "NWY",
    42: "DEN",
    43: "KIE",
    44: "BER",
    45: "PRU",
    46: "SIL",
    47: "MUN",
    48: "RUH",
    49: "HOL",
    50: "BEL",
    51: "PIC",
    52: "BRE",
    53: "PAR",
    54: "BUR",
    55: "MAR",
    56: "GAS",
    57: "BAR",
    58: "NWG",
    59: "NTH",
    60: "SKA",
    61: "HEL",
    62: "BAL",
    63: "BOT",
    64: "NAO",
    65: "IRI",
    66: "ENG",
    67: "MAO",
    68: "WES",
    69: "LYO",
    70: "TYS",
    71: "ION",
    72: "ADR",
    73: "AEG",
    74: "EAS",
    75: "BLA",
    76: "TYR",
    77: "BOH",
    78: "VIE",
    79: "TRI",
    80: "BUD",
    81: "GAL",
}

COUNTRY_ID_TO_POWER_FRANCE_VS_AUSTRIA = {1: "FRANCE", 2: "AUSTRIA"}

COUNTRY_ID_TO_POWER_OR_ALL_FRANCE_VS_AUSTRIA = {0: "ALL", 1: "FRANCE", 2: "AUSTRIA"}

TERR_ID_TO_LOC_BY_MAP = {
    GameVariant.CLASSIC: TERR_ID_TO_LOC,
    GameVariant.FVA: TERR_ID_TO_LOC_FRANCE_VS_AUSTRIA,
}

COUNTRY_ID_TO_POWER_MY_MAP = {
    GameVariant.CLASSIC: COUNTRY_ID_TO_POWER,
    GameVariant.FVA: COUNTRY_ID_TO_POWER_FRANCE_VS_AUSTRIA,
}

COUNTRY_ID_TO_POWER_OR_ALL_MY_MAP = {
    GameVariant.CLASSIC: COUNTRY_ID_TO_POWER_OR_ALL,
    GameVariant.FVA: COUNTRY_ID_TO_POWER_OR_ALL_FRANCE_VS_AUSTRIA,
}

COUNTRY_POWER_TO_ID = {v: k for k, v in COUNTRY_ID_TO_POWER.items()}

# these are the 22 supply centers that should have units on turn 0
PROPER_START_TERR_IDS = {
    11,
    12,
    15,
    2,
    22,
    23,
    24,
    27,
    29,
    3,
    31,
    37,
    38,
    41,
    46,
    47,
    49,
    6,
    72,
    73,
    74,
    79,
}

DATASET_DRAW_MESSAGE = "Voted for Draw"
DEPRECATED_DATASET_DRAW_MESSAGE = "/draw"
DATASET_NODRAW_MESSAGE = "Un-voted for Draw"
PUBLIC_DRAW_TYPE = "draw-votes-public"
PRIVATE_DRAW_TYPE = "draw-votes-hidden"
DRAW_VOTE_TOKEN = "<DRAW>"
UNDRAW_VOTE_TOKEN = "<NODRAW>"


def group_by(collection, fn):
    """Group the elements of a collection by the results of passing them through fn
    Returns a dict, {k_0: list[e_0, e_1, ...]} where e are elements of `collection` and
    f(e_0) = f(e_1) = k_0
    """
    r = defaultdict(list)
    for e in collection:
        k = fn(e)
        r[k].append(e)
    return r


class GoodGameCheckException(Exception):
    """Raised for a game which does not pass some basic initial checks"""

    pass


def check_is_good_game(db, game_id) -> None:
    """Returns None for good games, error string for bad games"""
    moves = db.execute(
        f"""SELECT turn, terrID, type
               FROM {TABLE_MOVES}
               WHERE hashed_gameID=?
            """,
        (game_id,),
    ).fetchall()

    # Criterion: has non-hold moves in turn 0
    if all(typ == "Hold" for (turn, _, typ) in moves if int(turn) == 0):
        raise GoodGameCheckException(
            f"Found bad game {game_id}: does not have non-hold moves in the first turn"
        )

    # Criterion: is at least 5 turns long
    turns = {int(turn) for (turn, _, _) in moves}
    if not all(t in turns for t in range(5)):
        raise GoodGameCheckException("Found bad game {game_id}: does not have at least 5 turns")

    # Criterion: has proper starting units
    if {int(terr_id) for (turn, terr_id, _) in moves if int(turn) == 0} != set(
        PROPER_START_TERR_IDS
    ):
        raise GoodGameCheckException(f"Found bad game {game_id}: no proper starting units")

    # Meets all criteria: good game
    return None


def find_all_games(db, press_type=None):
    variant_id = int(GameVariant.CLASSIC)
    query = f"SELECT hashed_id FROM {TABLE_GAMES} WHERE variantID={variant_id} AND phase='Finished' AND playerTypes='Members'"
    if press_type is not None:
        query += f" AND pressType = '{press_type}"
    games = db.execute(query)
    return [x[0] for x in games]


def get_message_table_schema(db):
    cursor = db.execute(f"SELECT * FROM {TABLE_MESSAGES}")
    colnames = [x[0] for x in cursor.description]
    return colnames


def move_row_to_order_str(row):
    """Convert a db row from wD_MovesArchive into an order string
    Returns a 3-tuple:
      - turn
      - power string, e.g. "AUSTRIA"
      - order string, e.g. "A RUH - BEL"
    N.B. Some order strings are incomplete: supports, convoys, and destroys are
    missing the unit type, since it is not stored in the db and must be
    determined from the game context.
    e.g. this function will return "A PAR S BRE - PIC" instead of the proper
    "A PAR S A BRE - PIC", since it is unknown which unit type resides at BRE
    Similarly, this function may return "X BRE X" instead of "A BRE D" or "F BRE D"
    """

    (
        gameID,
        turn,
        terrID,
        countryID,
        unitType,
        success,
        dislodged,
        typ,
        toTerrID,
        fromTerrID,
        viaConvoy,
    ) = row

    power = COUNTRY_ID_TO_POWER[int(countryID)]  # e.g. "ITALY"
    loc = TERR_ID_TO_LOC[int(terrID)]  # short code, e.g. "TRI"

    if typ == "Build Army":
        return turn, power, "A {} B".format(loc)
    if typ == "Build Fleet":
        return turn, power, "F {} B".format(loc)
    if typ == "Destroy":
        return turn, power, "X {} X".format(loc)

    unit_type = unitType[0].upper()  # "A" or "F"

    if typ == "Hold":
        return turn, power, "{} {} H".format(unit_type, loc)
    elif typ == "Move":
        to_loc = TERR_ID_TO_LOC[int(toTerrID)]
        via_suffix = " VIA" if viaConvoy == "Yes" else ""
        return turn, power, "{} {} - {}{}".format(unit_type, loc, to_loc, via_suffix)
    elif typ == "Support hold":
        to_loc = TERR_ID_TO_LOC[int(toTerrID)]
        return turn, power, "{} {} S {}".format(unit_type, loc, to_loc)
    elif typ == "Support move":
        from_loc = TERR_ID_TO_LOC[int(fromTerrID)]
        to_loc = TERR_ID_TO_LOC[int(toTerrID)]
        return (turn, power, "{} {} S {} - {}".format(unit_type, loc, from_loc, to_loc))
    elif typ == "Convoy":
        from_loc = TERR_ID_TO_LOC[int(fromTerrID)]
        to_loc = TERR_ID_TO_LOC[int(toTerrID)]
        return (turn, power, "{} {} C {} - {}".format(unit_type, loc, from_loc, to_loc))
    elif typ == "Retreat":
        to_loc = TERR_ID_TO_LOC[int(toTerrID)]
        return turn, power, "{} {} R {}".format(unit_type, loc, to_loc)
    elif typ == "Disband":
        return turn, power, "{} {} D".format(unit_type, loc)
    else:
        raise ValueError(
            "Unexpected move type = {} in hashed_gameID = {}, turn = {}, terrID = {}".format(
                typ, gameID, turn, terrID
            )
        )


def get_game_orders(db, game_id):
    """Return a dict mapping turn -> list of (turn, power, order) tuples
    i.e. return type is Dict[int, List[Tuple[int, str, str]]]
    """
    # gather orders
    turn_power_orders = [
        move_row_to_order_str(row)
        for row in db.execute(f"SELECT * FROM {TABLE_MOVES} WHERE hashed_gameID=?", (game_id,))
    ]
    orders_by_turn = group_by(turn_power_orders, lambda tpo: tpo[0])
    orders_by_turn = {int(k): v for k, v in orders_by_turn.items()}

    # major weirdness in the db: if the game ends in a draw, the orders from
    # the final turn are repeated, resulting in a bunch of invalid orders.
    game_over, last_turn = db.execute(
        f"SELECT gameOver,turn FROM {TABLE_GAMES} WHERE hashed_id=?", (game_id,)
    ).fetchone()
    last_turn = int(last_turn)
    if game_over == "Drawn":
        last_orders = {(power, order) for (_, power, order) in orders_by_turn[last_turn]}
        penult_orders = {(power, order) for (_, power, order) in orders_by_turn[last_turn - 1]}
        if last_orders == penult_orders:
            # fix this weirdness by removing the duplicate orders
            del orders_by_turn[last_turn]

    return orders_by_turn


def get_game_messages_by_turn(message_json):
    """
    Organizes a list of game messages by turn ID
    """
    game_messages_by_turn = {}
    for message in message_json:
        turn = int(message["turn"])
        game_messages_by_turn.setdefault(turn, [])
        game_messages_by_turn[turn].append(message)

    return game_messages_by_turn


def process_game(db, game_id, message_json, log_path=None):
    """Search db for moves from `game_id` and process them through a Game()
    Return a Game object with all moves processed
    """

    # gather orders
    orders_by_turn = get_game_orders(db, game_id)

    # gather messages if they exist
    if message_json is not None:
        message_json.sort(key=lambda msg: int(msg["timeSent"]))
        messages_by_turn = get_game_messages_by_turn(message_json)
        unprocessed_message_turns = list(messages_by_turn.keys())

        missing_message_turns = 0
        for turn in range(len(orders_by_turn)):
            if turn not in unprocessed_message_turns:
                missing_message_turns += 1
        if missing_message_turns > 0:
            logging.getLogger("p").warning(
                f"Game {game_id} missing messages for {missing_message_turns} / {len(orders_by_turn)} turns"
            )

    # run them through a diplomacy.Game
    game = Game(is_full_press=get_is_full_press(db, game_id))
    game.set_metadata("draw_type", get_game_draw_type(db, game_id))

    for turn in range(len(orders_by_turn)):
        # separate orders into one of {"MOVEMENT", "RETREATS", "DISBANDS", "BUILDS"}
        orders_by_category = group_by(orders_by_turn[turn], lambda tpo: get_order_category(tpo[2]))

        messages_by_phase = {}
        if message_json is not None:
            if turn in messages_by_turn:
                turn_messages = messages_by_turn[turn]
                turn_messages = sorted(turn_messages, key=lambda x: int(x["timeSent"]))
                messages_by_phase = get_messages_by_phase(turn_messages, log_game_id=game_id)
                unprocessed_message_turns.remove(turn)

        logging.getLogger("p").debug("=======================> TURN {}".format(turn))
        logging.getLogger("p").debug(
            "Turn orders from db: {}".format(pformat(orders_by_turn[turn]))
        )

        turn_r_phase = game.current_short_phase[:-1] + "R"
        turn_a_phase = "W" + game.current_short_phase[1:-1] + "A"

        # process movements
        if game.current_short_phase in messages_by_phase:
            set_phase_messages(game, messages_by_phase[game.current_short_phase])

        set_phase_orders(game, orders_by_category["MOVEMENT"])
        logging.getLogger("p").debug("process {}".format(game.phase))
        game.process()
        logging.getLogger("p").debug(
            "post-process units: {}".format(pformat(game.get_state()["units"]))
        )

        # process retreat phase messages -- note that the Game object may be
        # passed the R-phase (e.g. if there are no legal retreats) but webdip
        # may have let users message during this phase anyway, so we add those
        # messages to the next phase
        if turn_r_phase in messages_by_phase:
            set_phase_messages(game, messages_by_phase[turn_r_phase])

        # process all retreats
        if game.phase.split()[-1] == "RETREATS":
            set_phase_orders(game, orders_by_category["RETREATS"])

            # which locs require a move?
            orderable_locs = {
                loc for locs in game.get_orderable_locations().values() for loc in locs
            }

            # which locs have been ordered already?
            ordered_locs = {
                order.split()[1].split("/")[0]
                for orders in game.get_orders().values()
                for order in orders
            }

            # which locs are missing a move?
            missing_locs = orderable_locs - ordered_locs
            logging.getLogger("p").debug("retreat phase missing locs: {}".format(missing_locs))

            # if there is a disband for this loc, process it in the retreat phase
            for loc in missing_locs:
                power, order = pop_order_at_loc(orders_by_category["DISBANDS"], loc)
                if order is not None:
                    logging.getLogger("p").debug("set order {} {}".format(power, [order]))
                    set_phase_orders(game, [(turn, power, order)])

            # don't process if we are in the game over phase and there are no
            # orders to process this phase
            if turn < len(orders_by_turn) - 1 or any(game.get_orders().values()):
                logging.getLogger("p").debug("process {}".format(game.phase))
                game.process()

            logging.getLogger("p").debug(
                "post-process units: {}".format(pformat(game.get_state()["units"]))
            )

        # process A-phase messages -- note that the Game object may be
        # passed the A-phase (e.g. if there are no legal moves) but webdip
        # may have let users message during this phase anyway, so we add those
        # messages to the next phase
        if turn_a_phase in messages_by_phase:
            set_phase_messages(game, messages_by_phase[turn_a_phase])

        # process builds, remaining disbands
        if game.phase.split()[-1] == "ADJUSTMENTS":
            set_phase_orders(game, orders_by_category["BUILDS"] + orders_by_category["DISBANDS"])
            logging.getLogger("p").debug("process {}".format(game.phase))

            # don't process if we are in the game over phase and there are no
            # orders to process this phase
            if turn < len(orders_by_turn) - 1 or any(game.get_orders().values()):
                game.process()

            logging.getLogger("p").debug(
                "post-process units: {}".format(pformat(game.get_state()["units"]))
            )

    if message_json is not None and len(unprocessed_message_turns) > 0:
        # add any post game messages
        for msg_turn in unprocessed_message_turns:
            if msg_turn > turn:
                # add post game messages
                turn_messages = messages_by_turn[msg_turn]
                set_phase_messages(game, turn_messages)
            else:
                logging.getLogger("p").warn(
                    f"Unprocessed message turn {msg_turn} for game {game_id} (last game turn was {turn})"
                )

    game = maybe_process_dangling_retreat_phase(game, game_id=game_id)

    # Run some assertions before returning game object, removing this game from
    # the dataset if the assertions fail
    if message_json:
        # Check that all valid messages were added to game object before returning
        sorted_game_message_objs = sorted(
            [m for ms in game.message_history.values() for m in ms.values()]
            + [m for m in game.messages.values()],
            key=lambda m: int(m["time_sent"]),
        )
        game_messages = {m["message"] for m in sorted_game_message_objs}
        missing_messages = [
            m
            for i, m in enumerate(message_json)
            if is_valid_message(m)
            and m["message"] not in game_messages
            and not is_noncontiguous_message(message_json, i)
        ]
        assert len(missing_messages) == 0, f"Missing messages in game {game_id}"

        # Check that messages are in order
        sorted_json_message_objs = [
            m
            for i, m in enumerate(sorted(message_json, key=lambda m: int(m["timeSent"])))
            if is_valid_message(m) and not is_noncontiguous_message(message_json, i)
        ]
        assert [m["message"] for m in sorted_game_message_objs] == [
            m["message"] for m in sorted_json_message_objs
        ], f"Messages added out of order in game {game_id}"

    return game


def maybe_process_dangling_retreat_phase(game: Game, game_id=None) -> Game:
    """
    Check if a game was abandoned before the final (meaningless) retreat phase
    was played, and process that phase with fake disband orders.

    This occurs because humans often don't finish games where they have lost,
    even though the game technically does not count your supply centers as lost
    until after the retreat phase.
    """
    if game.current_short_phase[-1] != "R":
        return game
    game_bak = Game(game)
    for power, locs in game.get_orderable_locations().items():
        # disband any unit that has been dislodged
        orders = [game.get_all_possible_orders()[loc][-1] for loc in locs]
        if orders:
            game.set_orders(power, orders)
    game.process()

    # if this results in a completed game, then return the game with this fake
    # retreat phase
    if game.phase == "COMPLETED":
        logging.debug(f"maybe_process_dangling_retreat_phase completed {game_id}")
        return game
    else:
        assert game_bak.current_short_phase[-1] == "R"
        return game_bak


def get_messages_by_phase(turn_messages, *, log_game_id=None):

    assert len(set(m["turn"] for m in turn_messages)) == 1, "Messages must be from same turn"
    turn = turn_messages[0]["turn"]

    turn_messages.sort(key=lambda msg: int(msg["timeSent"]))

    # Set default "last_phase" if first msg has Unknown phase
    last_phase = turn_to_phase(int(turn_messages[0]["turn"]), "Diplomacy")
    assert last_phase is not None

    messages_by_phase = {}
    expired_phases = set()
    n_expired_phase_messages = 0
    for msg in turn_messages:
        # If phase is unknown, use previous message phase
        phase = turn_to_phase(int(msg["turn"]), msg["phase"]) or last_phase
        if phase in expired_phases:
            logging.getLogger("p").warning(
                f"Phase not contiguous! game_id={log_game_id} turn={turn}"
            )
            n_expired_phase_messages += 1
            # Do not include non-contiguous phase message
            continue
        if phase != last_phase:
            expired_phases.add(last_phase)
            last_phase = phase
        messages_by_phase.setdefault(phase, [])
        messages_by_phase[phase].append(msg)

    assert n_expired_phase_messages <= 1, "Too many expired phase messages, skipping game"

    return messages_by_phase


def is_valid_message(message: Dict):
    from_id = int(message["fromCountryID"])
    to_id = int(message["toCountryID"])
    assert from_id in COUNTRY_ID_TO_POWER_OR_ALL
    assert to_id in COUNTRY_ID_TO_POWER_OR_ALL

    # This skips GameMaster messages but preserves player->public messages
    if from_id == 0:
        return False
    return True


def set_phase_messages(game, phase_messages):
    for message in sorted(phase_messages, key=lambda x: int(x["timeSent"])):
        if not is_valid_message(message):
            continue
        sender = COUNTRY_ID_TO_POWER_OR_ALL[int(message["fromCountryID"])]
        recipient = COUNTRY_ID_TO_POWER_OR_ALL[int(message["toCountryID"])]

        if message["message"] == DATASET_DRAW_MESSAGE and sender == recipient:
            game.set_metadata("has_draw_votes", "True")
            message["message"] = DRAW_VOTE_TOKEN
            if game.get_metadata("draw_type") == PUBLIC_DRAW_TYPE:
                recipient = "ALL"

        if message["message"] == DATASET_NODRAW_MESSAGE and sender == recipient:
            game.set_metadata("has_draw_votes", "True")
            message["message"] = UNDRAW_VOTE_TOKEN
            if game.get_metadata("draw_type") == PUBLIC_DRAW_TYPE:
                recipient = "ALL"

        # Fix deprecated draw votes
        if message["message"] == DEPRECATED_DATASET_DRAW_MESSAGE and recipient == "ALL":
            game.set_metadata("has_draw_votes", "True")
            message["message"] = DRAW_VOTE_TOKEN

        game.add_message(
            sender,
            recipient,
            message["message"],
            Timestamp.from_seconds(message["timeSent"]),
            increment_on_collision=True,
        )


def set_phase_orders(game: Game, phase_orders: List[Tuple[int, Power, str]]):
    logging.getLogger("p").debug("set_phase_orders start {}".format(game.phase))

    # map of loc -> (power, "A/F")
    unit_at_loc = {}
    for power, unit_list in game.get_state()["units"].items():
        for unit in unit_list:
            unit_type, loc = unit.strip("*").split()
            unit_at_loc[loc] = (power, unit_type)
            unit_at_loc[loc.split("/")[0]] = (power, unit_type)

    orders_by_power = group_by(phase_orders, lambda tpo: tpo[1])
    for power, tpos in orders_by_power.items():
        # compile orders, adding in missing unit type info for supports/convoys/destroys
        orders = []
        for _, _, order in tpos:
            split = order.split()

            # fill in unit type for supports / convoys
            if split[2] in ("S", "C"):
                loc = split[3]
                _, unit_type = unit_at_loc[loc]
                split = split[:3] + [unit_type] + split[3:]

            # fill in unit type for destroys
            elif split[0] == "X":
                loc = split[1]
                _, unit_type = unit_at_loc[loc]
                split = [unit_type, loc, "D"]

            possible_orders = set(game.get_all_possible_orders()[split[1]])
            if " ".join(split) in possible_orders:
                orders.append(" ".join(split))
            else:
                # if order is not valid, try location coastal variants, since
                # some orders are coming out of the db without the proper
                # coast.
                variant_split = get_valid_coastal_variant(split, possible_orders)
                if variant_split is not None:
                    orders.append(" ".join(variant_split))
                else:
                    # if there are no valid coastal variants, check if this is
                    # a disband that has already been processed. This sometimes
                    # happens when a unit is dislodged and has nowhere to
                    # retreat -- there will be a disband in the db, but the
                    # Game object disbands it automatically.
                    if split[2] == "D" and unit_at_loc.get(split[1], (None, None))[0] != power:
                        logging.getLogger("p").debug(
                            'Skipping disband: {} "{}"'.format(power, order)
                        )
                    else:
                        error_msg = 'Bad order: {} "{}", possible_orders={}'.format(
                            power, order, possible_orders
                        )
                        logging.getLogger("p").error(error_msg)
                        err = ValueError(error_msg)
                        err.partial_game = game
                        raise err

        # ensure that each order is valid
        for order in orders:
            loc = order.split()[1]
            assert order in game.get_all_possible_orders()[loc], (
                game.phase,
                (power, loc, order),
                game.get_all_possible_orders()[loc],
            )

        logging.getLogger("p").debug('set_phase_orders -> {} "{}"'.format(power, orders))
        game.set_orders(power, orders)


def get_valid_coastal_variant(split, possible_orders):
    """Find a variation on the `split` order that is in `possible_orders`
    Args:
        - split: a list of order string components,
                 e.g. ["F", "AEG", "S", "F", "BUL", "-", "GRE"]
        - possible_orders: a list of order strings,
                e.g. ["F AEG S F BUL/SC - GRE", "F AEG H", "F AEG - GRE", ...]
    This function tries variations (e.g. "BUL", "BUL/SC", etc.) of the `split`
    order until one is found in `possible_orders`.
    Returns a split order, or None if none is found
    e.g. for the example inputs above, this function returns:
            ["F", "AEG", "S", "F", "BUL/SC", "-", "GRE"]
    """
    for idx in [1, 4, 6]:  # try loc, from_loc, and to_loc
        if len(split) <= idx:
            continue
        for variant in [split[idx].split("/")[0] + x for x in ["", "/NC", "/EC", "/SC", "/WC"]]:
            try_split = split[:idx] + [variant] + split[(idx + 1) :]
            if " ".join(try_split) in possible_orders:
                return try_split
    return None


def pop_order_at_loc(tpos, loc):
    """If there is an order at loc in tpos, remove and return it
    tpos: A list of (turn, power, order) tuples
    Returns: (power, order) if found, else (None, None)
    """
    for i, (_, power, order) in enumerate(tpos):
        order_loc = order.split()[1]
        if loc.split("/")[0] == order_loc.split("/")[0]:
            del tpos[i]
            return (power, order)
    return None, None


def get_order_category(order):
    """Given an order string, return the category type, one of:
    {"MOVEMENT, "RETREATS", "DISBANDS", "BUILDS"}
    """
    order_type = order.split()[2]
    if order_type in ("X", "D"):
        return "DISBANDS"
    elif order_type == "B":
        return "DISBANDS"
    elif order_type == "R":
        return "RETREATS"
    else:
        return "MOVEMENT"


def is_noncontiguous_message(message_json, i):
    """
    There is a bug in the webdip data where some turns have a "Diplomacy"-phase
    message which occurs chronologically after a non-"Diplomacy"-phase message
    in the same turn (which should never happen).

    This function return True if the i-th message in message_json meets this condition.
    """
    if i == 0:
        # First message can't be noncontiguous
        return False
    msg = message_json[i]
    prev = message_json[i - 1]
    return (
        msg["turn"] == prev["turn"]
        and msg["timeSent"] > prev["timeSent"]
        and msg["phase"] == "Diplomacy"
        and prev["phase"] != "Diplomacy"
    )


def load_messages(db, sql_game_id: int) -> List[Dict]:
    """
    Loads messages for a particular game.
    """
    message_table_schema = get_message_table_schema(db)

    def _convert_message_row(row):
        return {message_table_schema[i]: x for i, x in enumerate(row)}

    messages_raw = db.execute(
        f"""SELECT *
        FROM {TABLE_MESSAGES}
        WHERE hashed_gameID=?
        """,
        (sql_game_id,),
    ).fetchall()

    messages_data = [_convert_message_row(x) for x in messages_raw]
    return messages_data


def get_is_full_press(db, sql_game_id: int) -> bool:
    (press_type,) = db.execute(
        f"SELECT pressType FROM {TABLE_GAMES} WHERE hashed_id=?", (sql_game_id,)
    ).fetchone()
    return press_type != "NoPress"


def get_game_draw_type(db, sql_game_id: int) -> str:
    (draw_type,) = db.execute(
        f"SELECT drawType FROM {TABLE_GAMES} WHERE hashed_id=?", (sql_game_id,)
    ).fetchone()
    assert draw_type in [PUBLIC_DRAW_TYPE, PRIVATE_DRAW_TYPE], draw_type
    return draw_type


def process_and_save_game(
    sql_game_id: int, db_path: str, out_dir: str
) -> Tuple[Optional[int], int, Optional[int]]:
    """
    Process and save game.

    - sql_game_id: Game ID of the game in the database
    - db_path: Path to the database

    Returns a tuple of [HashID, SQL Game ID, and JSON Game ID]
    - HashID is unique to each game JSON
    - SQL Game ID is the game ID as it is listed in the SQL
        database specified in db_path
    - JSON Game ID is the game ID as it will be written to file,
        which is preserved across versions
    """
    retval = (None, sql_game_id, None)
    db = sqlite3.connect(db_path)

    if out_dir is not None:
        log_path = os.path.join(out_dir, "game_new_id={}.log".format(sql_game_id))
        if not args.overwrite and os.path.isfile(log_path):
            return retval
        logging.getLogger("p").propagate = False
        logging.getLogger("p").setLevel(logging.DEBUG)
        logging.getLogger("p").handlers = [logging.FileHandler(log_path)]
    else:
        log_path = None

    try:
        check_is_good_game(db, sql_game_id)
        message_json = load_messages(db, sql_game_id)
        game = process_game(db, sql_game_id, message_json, log_path=log_path)
    except Exception as err:
        logging.getLogger("p").exception(
            "Exception processing game with new ID {}".format(sql_game_id)
        )
        with open(os.path.join(out_dir, f"game_{sql_game_id}.exc"), "w") as f:
            f.write(traceback.format_exc())
        if hasattr(err, "partial_game"):
            # partial save path
            partial_save_path = os.path.join(out_dir, "game_new_id={}.partial".format(sql_game_id))
            with open(partial_save_path, "w") as stream:
                stream.write(err.partial_game.to_json())
            logging.debug("Saved partial game to {}".format(partial_save_path))
        return retval

    # find the hashed ID to map back to the old ID
    game_hash = game.compute_order_history_hash()
    # load previous data hash
    with open(args.previous_data_hash, "r") as f:
        previous_data_hash = [x.split(" ") for x in f.read().splitlines() if "None" not in x]
        previous_data_hash = {int(k): int(v) for k, v in previous_data_hash}
    json_game_id = previous_data_hash.get(game_hash)
    if json_game_id is None:
        logging.info(f"Found new game: {json_game_id}")
        json_game_id = max(previous_data_hash.values()) + sql_game_id

    # save path
    save_path = os.path.join(out_dir, "game_{}.json".format(json_game_id))
    with open(save_path, "w") as stream:
        stream.write(game.to_json())

    logging.info("Saved to {}".format(save_path))
    return game_hash, sql_game_id, json_game_id


def write_new_data_hash_fle(
    out_dir: str, hash_sqlid_jsonid_tups: List[Tuple[Optional[int], int, Optional[int]]]
):
    """
    Write new data hash files, to preserve for future data dumps.
    """
    new_hash_fle = os.path.join(out_dir, "hash_to_game_id.txt")
    new_hash_map_fle = os.path.join(out_dir, "hash_to_sqlid_to_jsonid.txt")
    logging.info(f"Saving hash information to file: {new_hash_fle}")
    logging.info(
        f"For debugging purposes, also saving a map of new ID (SQL table) to old ID (game JSON) to: {new_hash_map_fle}"
    )
    with open(new_hash_fle, "w") as f:
        with open(new_hash_map_fle, "w") as g:
            for hash, sqlid, jsonid in hash_sqlid_jsonid_tups:
                f.write(f"{hash} {jsonid}\n")
                g.write(f"{hash} {sqlid} {jsonid}\n")


def symlink_full_press(out_dir: str):
    """
    Find games that contain messages and separate them to a new folder.
    """

    def contains_messages(game_path: str) -> bool:
        """
        Returns True or False depending whether a game contains any messages at all
        """
        with open(game_path, "r") as f:
            game = json.load(f)

        for phase in game["phases"]:
            if phase["messages"]:
                return True

        return False

    full_press_folder = out_dir.replace("all_games", "full_press_games")
    if not os.path.exists(full_press_folder):
        os.makedirs(full_press_folder)

    logging.info(f"Symlinking full press games to folder: {full_press_folder}")

    total = 0
    full_press_games = 0
    for game_json in tqdm(glob(os.path.join(out_dir, "game_*.json"))):
        total += 1
        original_path = os.path.join(out_dir, game_json)
        full_press = contains_messages(original_path)
        if full_press:
            full_press_games += 1
            syml = game_json.replace("all_games", "full_press_games")
            if not os.path.exists(syml):
                os.symlink(original_path, syml)

    logging.info(f"Found {full_press_games} full press games out of {total} total games")


if __name__ == "__main__":
    STDOUT = "/tmp/debug.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[logging.FileHandler(STDOUT), logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Dump game.json files to this dir")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process and overwrite existing game.json files",
    )
    parser.add_argument(
        "--db-path", help="Path to SQLITE db file", required=True,
    )
    parser.add_argument(
        "--previous-data-hash",
        required=True,
        help=(
            "Path to previous data hash file: this is required to be able to preserve "
            "game IDs between versions. The hash file should always be 1 data version behind "
            "It is automatically computed for the current version at the conclusion of this script. "
        ),
    )
    parser.add_argument(
        "--limit-n-games", type=int, default=None, help="Process only the first N games found"
    )
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    logging.info(f"Writing stdout to file: {STDOUT}")
    logging.info("Ensuring out dir exists: {}".format(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    all_games_out_dir = os.path.join(args.out_dir, "all_games")
    os.makedirs(all_games_out_dir, exist_ok=True)

    db = sqlite3.connect(args.db_path)
    all_sql_game_ids = sorted(list(find_all_games(db)), key=lambda x: int(x))
    if args.limit_n_games:
        all_sql_game_ids = all_sql_game_ids[: args.limit_n_games]
    logging.info(f"Found {len(all_sql_game_ids)} finished, non-bot games...")

    if args.parallel:
        ret = joblib.Parallel(n_jobs=-1, verbose=1)(
            joblib.delayed(process_and_save_game)(sql_game_id, args.db_path, all_games_out_dir)
            for sql_game_id in tqdm(all_sql_game_ids)
        )
    else:
        ret = []
        for sql_game_id in all_sql_game_ids:
            retval = process_and_save_game(sql_game_id, args.db_path, all_games_out_dir)
            ret.append(retval)

    logging.info(f"Initial game processing completed. STDOUT written to file: {STDOUT}")

    # Hash managing
    write_new_data_hash_fle(args.out_dir, ret)

    # Symlink full press games
    symlink_full_press(all_games_out_dir)
