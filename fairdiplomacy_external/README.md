# fairdiplomacy_external

### License

We use small parts of https://github.com/diplomacy/diplomacy for two tasks:

1) Rendering of Diplomacy game board maps

2) Formatting of order submission objects in the manner that https://webdiplomacy.net/ expects them.

The aforementioned repository is licensed under the APGLv3 License, which requires that any projects that use it must also be licensed under the APGLv3 License. Therefore, we separate those parts of our code referencing this repository into this directory fairdiplomacy_external, and we solely license it under APGLv3, separately from the rest of this GitHub repository, which uses an MIT license.

See the [LICENSE](LICENSE) file for details.

### Downloading HTML renderings of Cicero's experiment games

As mentioned in the top-level README, JSON data for games that Cicero played in are located in `data/cicero_redacted_games`. These games have also been converted to human-readable HTML visualisations with `fairdiplomacy_external/game_to_html.py` and can be downloaded from [https://dl.fbaipublicfiles.com/diplomacy_cicero/games.tar.gz](https://dl.fbaipublicfiles.com/diplomacy_cicero/games.tar.gz).

### Installation
In addition to the installation directions for the rest of this repo as documented in the top-level README, using the code here also requires the requirements.txt in this directory to be installed via pip:

```
conda activate diplomacy_cicero
pip install -r fairdiplomacy_external/requirements.txt
```

### Visualizing games
We provide a script `fairdiplomacy_external/game_to_html.py` to render game jsons with HTML. Here is an example command:

`python fairdiplomacy_external/game_to_html.py data/example_fullpress_game.json -o game.html`

### Running agents on webdiplomacy (or a private webdiplomacy instance)
We provide here some code to run agents on [webdiplomacy.net](webdiplomacy.net). The script `fairdiplomacy_external/webdip_api.py` requires some configuration. In particular, it requires installation of Redis. Then, in `fairdiplomacy.webdip.message_approval_cache_api`, please update `REDIS_IP` and `PORT` global variables accordingly (so that they point to your Redis server). Then, you need to request an API key from the [webdiplomacy.net](webdiplomacy.net) team. Then, you can kick off a bot as follows. It will service all games under the specified account name / API key in round-robin fashion.

```
python run.py --adhoc -c conf/c07_play_webdip/play.prototxt \
   api_key=replace_this_with_api_key \
   account_name=replace_this_with_name_of_bot \
   allow_dialogue=true \
   log_dir=/path/to/where/you/want/logs/to/be/written \
   only_bump_msg_reviews_for_same_power=true \
   require_message_approval=false \
   is_backup=false \
   retry_exception_attempts=10 \
   reset_bad_games=1 \
   I.agent=agents/bqre1p_parlai_20220819_cicero_2.prototxt
```

