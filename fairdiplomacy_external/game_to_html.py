#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the APGLv3 license found in the
# LICENSE file in the fairdiplomacy_external directory of this source tree.
#
import argparse
from pathlib import Path
import heyhi
from typing import Optional
from fairdiplomacy import pydipcc
import os

import jinja2

import conf.misc_cfgs
import fairdiplomacy.game
from fairdiplomacy.models.consts import POWERS
import fairdiplomacy_external.ui

INDEX_HTML = (
    """
<!doctype html>
<html>
<head profile="http://www.w3.org/2005/10/profile">
"""
    + fairdiplomacy_external.ui.render.get_header_stuff()
    + """
</head>

<body>
{{ rendered_message_filterer|safe }}
{{ test_situation_saver|safe }}
<div class="container" style="margin: 10px">
    <h3>{{ title }}</h3>
    <div>
    <h2>Jump to phase</h2>
    {%for i in range(num_phases)%}
    {% if i > 0 and i < num_phases - 1 and phase_names[i - 1][1:5] != phase_names[i][1:5] %}</div><div>{% endif %}
    <a href="#{{phase_names[i]}}">{{phase_names[i]}}</a>
    {% endfor %}
    </div>
    <p/>
    {% for i in range(num_phases) %}
    <div>
        <a name="{{phase_names[i]}}"><h2>{{phase_names[i]}}</h2></a>
        {{ rendered_phases[i]|safe }}
    </div>
    <hr/>
    {% endfor %}
</div>
</body>
</html>
"""
)


def game_to_html(
    game: pydipcc.Game,
    title: str = "",
    annotations: Optional[conf.misc_cfgs.AnnotatedGame] = None,
    filter1: bool = False,
):
    rendered_message_filterer = fairdiplomacy_external.ui.render.render_message_filterer(
        fixed=True, filter1=filter1
    )
    phase_names = [p.name for p in game.get_phase_history()]
    rendered_phases = [
        fairdiplomacy_external.ui.render.render_phase(game, phase, annotations)
        for phase in phase_names
    ]
    num_phases = len(phase_names)
    test_situation_saver = fairdiplomacy_external.ui.render.render_test_situation_saver()
    template = jinja2.Template(INDEX_HTML)
    return template.render(**locals(), POWERS=POWERS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("-o", default="game.html")
    parser.add_argument("--title", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument(
        "--filter1",
        default=False,
        action="store_true",
        help="Only provide one power to filter by (instead of two)",
    )
    args = parser.parse_args()

    with open(args.json_path) as f:
        game = pydipcc.Game.from_json(f.read())

    if args.annotations is not None:
        annotations = heyhi.load_config(Path(args.annotations))
    else:
        annotations = None

    if args.title is not None:
        title = args.title
    else:
        title = os.path.abspath(args.json_path)
    html = game_to_html(game, title=title, annotations=annotations, filter1=args.filter1)

    with open(args.o, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
