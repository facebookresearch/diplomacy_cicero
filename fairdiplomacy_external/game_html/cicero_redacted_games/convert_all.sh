#!/bin/bash
set -e

# This script converts all the redacted cicero games to html visualizations.

B=$(dirname $0)/../../..
echo $B
for F in $B/data/cicero_redacted_games/*.json; do
    # The filenames are of the form "game_{game_id}_{power}_{users}.json"
    # strip the .json extension and the parent path
    F=$(basename $F .json)

    echo $F
    # split the filename into parts by underscore
    IFS=_ read -r -a PARTS <<< "$F"
    # get the game ID
    GAME_ID=${PARTS[1]}
    # get the power
    POWER=${PARTS[2]}
    # get the users
    USERS=${PARTS[3]}

    # do the equivalent of ",".join(USERS)
    USERS=$(echo $USERS | sed 's/./&,/g' | sed 's/,$//')
    python $B/fairdiplomacy_external/game_to_html.py data/cicero_redacted_games/$F.json \
        -o $B/fairdiplomacy_external/game_html/cicero_redacted_games/$F.html \
        --filter1 \
        --title "Game $GAME_ID. Cicero is $POWER . Dialogue with $USERS shown."
done
