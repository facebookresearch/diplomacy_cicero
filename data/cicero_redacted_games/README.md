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
They are also publicly hosted on S3 (list below).

## Publicly Hosted Game JSONs.
[game_433761_ENGLAND_AG](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_433761_ENGLAND_AG.html)

[game_433762_ITALY_AEFT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_433762_ITALY_AEFT.html)

[game_433920_RUSSIA_EI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_433920_RUSSIA_EI.html)

[game_433967_ENGLAND_IT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_433967_ENGLAND_IT.html)

[game_434015_GERMANY_AF](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_434015_GERMANY_AF.html)

[game_434119_AUSTRIA_FRT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_434119_AUSTRIA_FRT.html)

[game_434170_TURKEY_FIR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_434170_TURKEY_FIR.html)

[game_435086_FRANCE_RT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_435086_FRANCE_RT.html)

[game_435211_ITALY_EGT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_435211_ITALY_EGT.html)

[game_435440_ENGLAND_AR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_435440_ENGLAND_AR.html)

[game_435500_FRANCE_AI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_435500_FRANCE_AI.html)

[game_435605_RUSSIA_EFGT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_435605_RUSSIA_EFGT.html)

[game_436345_RUSSIA_AEFGIT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_436345_RUSSIA_AEFGIT.html)

[game_436958_RUSSIA_AEGIT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_436958_RUSSIA_AEGIT.html)

[game_437495_RUSSIA_IT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_437495_RUSSIA_IT.html)

[game_437697_ENGLAND_GR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_437697_ENGLAND_GR.html)

[game_437765_ITALY_EFR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_437765_ITALY_EFR.html)

[game_438141_FRANCE_EGR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_438141_FRANCE_EGR.html)

[game_439643_ENGLAND_T](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_439643_ENGLAND_T.html)

[game_439679_ITALY_R](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_439679_ITALY_R.html)

[game_439999_TURKEY_I](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_439999_TURKEY_I.html)

[game_441172_FRANCE_ERT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_441172_FRANCE_ERT.html)

[game_441488_ENGLAND_FGIT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_441488_ENGLAND_FGIT.html)

[game_443187_ITALY_FG](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_443187_ITALY_FG.html)

[game_443556_FRANCE_AI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_443556_FRANCE_AI.html)

[game_443777_ENGLAND_AIT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_443777_ENGLAND_AIT.html)

[game_443831_GERMANY_EI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_443831_GERMANY_EI.html)

[game_444322_AUSTRIA_ER](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_444322_AUSTRIA_ER.html)

[game_444400_TURKEY_G](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_444400_TURKEY_G.html)

[game_444641_RUSSIA_AFI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_444641_RUSSIA_AFI.html)

[game_445034_RUSSIA_AFGIT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_445034_RUSSIA_AFGIT.html)

[game_445608_FRANCE_R](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_445608_FRANCE_R.html)

[game_445663_RUSSIA_A](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_445663_RUSSIA_A.html)

[game_446643_AUSTRIA_GR](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_446643_AUSTRIA_GR.html)

[game_447229_ENGLAND_GI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_447229_ENGLAND_GI.html)

[game_447784_ITALY_RT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_447784_ITALY_RT.html)

[game_448105_ENGLAND_AG](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_448105_ENGLAND_AG.html)

[game_448741_ITALY_ERT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_448741_ITALY_ERT.html)

[game_449062_TURKEY_AFI](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_449062_TURKEY_AFI.html)

[game_449418_ENGLAND_GRT](https://dl.fbaipublicfiles.com/diplomacy_cicero/games/game_449418_ENGLAND_GRT.html)

