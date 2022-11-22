#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
""" Renderer
    Calls out to external renderer.
"""
import json
from typing import Optional

from fairdiplomacy import pydipcc

######

# This import triggers a GPL license for this file.
# As a result, we release the fairdiplomacy_external subdirectory under a GPL license,
# while releasing the rest of the repository under an MIT license.
import diplomacy.engine.renderer
import diplomacy.utils.export

######


def render(
    game: pydipcc.Game, phase: Optional[str] = None, incl_abbrev=True, hide_orders: bool = False
):
    if phase is not None:
        game = game.rolled_back_to_phase_end(phase)
    if hide_orders:
        game = pydipcc.Game(game)
        game.set_all_orders({})

    pygame = diplomacy.utils.export.from_saved_game_format(json.loads(game.to_json()))

    renderer = diplomacy.engine.renderer.Renderer(pygame)
    result: str = renderer.render(incl_abbrev=incl_abbrev)
    return result
