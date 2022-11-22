#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
from IPython.display import SVG, display
from fairdiplomacy.pydipcc import Game
from termcolor import colored

from fairdiplomacy_external.ui import map_renderer

POWER_COLORS = {
    "ENGLAND": "magenta",
    "FRANCE": "blue",
    "GERMANY": "yellow",
    "ITALY": "green",
    "AUSTRIA": "red",
    "RUSSIA": "cyan",
    "TURKEY": "yellow",
    "ALL": None,
}


def color_power(power):
    return colored(f"{power:7s}", POWER_COLORS[power], attrs=["bold"])


def view_phase(game, phase=None, msg_pwr1="ALL", msg_pwr2="ALL"):
    if isinstance(game, str):
        with open(game) as gf:
            game = Game.from_json(gf.read())
    if phase is None:
        phase = game.current_short_phase
    image = map_renderer.render(game, phase)
    if phase == game.current_short_phase:
        messages = game.messages.values()
    else:
        messages = [
            list(p.messages.values()) for p in game.get_phase_history() if p.name == phase
        ][0]
    display(SVG(image))
    for msg in messages:
        if msg_pwr1 != "ALL" and msg["sender"] != msg_pwr1 and msg["recipient"] != msg_pwr1:
            continue
        if msg_pwr2 != "ALL" and msg["sender"] != msg_pwr2 and msg["recipient"] != msg_pwr2:
            continue
        print(
            f"{color_power(msg['sender'])} -> {color_power(msg['recipient'])} ({msg['time_sent']}): {msg['message']}"
        )
