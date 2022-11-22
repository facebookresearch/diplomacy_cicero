Each json file in this folder is the data for one game played by Cicero.
Only the dialogue with players who have consented to have their data released are contained
in these files. Each file name has the format `game_{game_id}_{cicero_power}_{initials_of_consenting_powers}.json`

The games can be loaded through pydipcc, as follows:
```
from fairdiplomacy.pydipcc import Game

game = Game.from_json(open("/path/to/game.json").read())
```

You can convert these games to html visualizations with `python game_to_html.py /path/to/game.json`.
We provide the script `fairdiplomacy_external/game_html/cicero_redacted_games/convert_all.sh` to convert them all.
