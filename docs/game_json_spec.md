# Diplomacy Game JSON Specification, Version 1.0

This doc outlines the JSON representation of a Diplomacy game.

A game json is a dictionary with the following top-level fields:

```
{
    "version": string                // version string
    "id": string,                    // unique id
    "is_full_press": bool,           // True if the game allowed messages between players
    "map": string,                   // "standard" or "fva"
    "scoring_system": {string, int}, // described in the Scoring section below
    "metadata": Dict[str, str],      // described in the Metadata section below
    "phases": List[Phase],           // described in the Phase section below
}
```

### Phase

A `Phase` is a json dict with the following fields:

```
{
    "name": string,                // short phase name, e.g. "S1901M", "W1905A", or "COMPLETED"
    "messages": Dict[int, Message] // map from timestsamp (in centiseconds) to Message object (described in the Message section below)
    "orders": Orders               // described in section Orders
    "state":                       // describe in section State
    "phase_start_time": int        // OPTIONAL, timestamp in centiseconds
}
```

Some clarifications on phase names:
- A completed game may have a final phase with `"name": "COMPLETED"` for the purpose of recording post-game messages, a `phase_start_time`, and/or a final `state`.
- All other phase names are exactly 6 characters long.
    - The first character indicates the season: either `S` (spring), `F` (fall), or `W` (winter)
    - The next four characters indicate the year, beginning in `1901`
    - The final character indicates the sub-phase: either `M` (movement) or `R` (retreats) in spring/fall, or `A` (adjustments) in winter.


### Message

A `Message` is a json dict with the following fields:

```
{
    "phase": string,                  // short phase name, e.g. "S1901M", "W1905A", or "COMPLETED"
    "sender": Power,                  // e.g. "AUSTRIA" or "FRANCE"
    "recipient": Union[Power, "ALL"], // e.g. same as above, or "ALL" to indicate a public message
    "time_sent": int,                 // timsestamp in centiseconds
    "message": string,                // the message body
}
```

### Orders

An `Orders` is a json dict which maps from `Power` to `List[Order]`. An example is shown below:

```
{
  "AUSTRIA": [ "A VIE - BUD", "F TRI - ALB", "A BUD - RUM" ],
  "ENGLAND": [ "F EDI - NTH", "F LON - ENG", "A LVP - WAL" ],
  "FRANCE": [ "F BRE - MAO", "A PAR - BUR", "A MAR - SPA" ],
  "GERMANY": [ "F KIE - DEN", "A MUN - RUH", "A BER - SIL" ],
  "ITALY": [ "F NAP - ION", "A ROM - APU", "A VEN H" ],
  "RUSSIA": [ "F STP/SC - BOT", "A MOS - UKR", "A WAR - GAL", "F SEV - BLA" ],
  "TURKEY": [ "F ANK - BLA", "A SMY - ARM", "A CON - BUL" ]
}
```

Some clarifications on `Order` strings:
- The ordering of the orders in a list does not matter; consumers of this format should be robust to any permutation.
- Retreat orders must be written "F BLA R SEV", not "F BLA - SEV"
- Armies attempting to move via convoy write their orders with the VIA suffix, e.g. "A LON - BRE VIA"
- Convoys are written without the VIA suffix, eg. "F NTH C A LON - BRE"
- The destination of a fleet move to BUL, STP, or SPA must specify the coast, even if only one is possible. e.g. "F LYO - SPA" is invalid, "F LYO - SPA/SC" is valid.

### State

A `State` is a json dict with the following fields:

```
{
    "name": string                           // short phase name, e.g. "S1901M", "W1905A", or "COMPLETED"
    "units": Dict[Power, List[Unit]]         // units controlled by each power
    "centers": Dict[Power, List[Loc]]        // supply centers controlled by each power
    "retreats": Dict[Power, Dict[Unit, Loc]] // valid locations to which each unit can retreat this phase
    "builds": Dict[Power, Builds]            // see Builds section below
}
```

During retreat phases, a unit that has been dislodged will be prefixed by an asterix in the `units` list.
In the following example, Turkey has dislodged France's fleet in `LYO`. France's fleet has only one location
to which it can retreat: `MAR`.

```
{
    "name": "F1904R",
    "units": {
        "FRANCE": ["F TUN", "*F LYO", ...],
        "TURKEY": ["A MOS", "F LYO", ...],
    },
    "retreats": {
        "F LYO": ["MAR"]
    }
}
```

#### Builds

A `Builds` object is a json dict with the following fields:

```
{
    "count": int        // if positive, number of builds that can be ordered this phase.
                        // if negative, number of disbands that must be ordered this phase.
    "homes": List[Loc]  // if "count" is positive, a list of home supply centers on which a build can be ordered
}
```

### Metadata

`Metadata` is a json dict with string keys and string values. All fields are
optional. Some fields that may be present in our game jsons:

```
    "has_draw_votes": "True" OR "False"
    "draw_type": "draw-votes-public" OR "draw-votes-hidden"
```

### Scoring

Represents the scoring system used in the case of a draw. In all scoring systems, the score is normalized to sum to 1. Only `sum_of_squares` and `draw_size` scoring are currently supported by dipcc. Strings for some other common scoring systems are reserved here, and more scoring systems may be added in the future.

For legacy compatibility, the int values 0 and 1 may be used instead of the strings `sum_of_squares` and `draw_size`, respectively.


- `"sum_of_squares"` or `0`: In case of solo, soloist scores 100% and others score 0%. Otherwise, score is divided proportional to (owned centers)^2.
- `"draw_size"` or `1`: In case of solo, soloist scores 100% and others score 0%. Otherwise, score is divided equally among surviving players.
- `"c_diplo_100"`: In case of solo, soloist scores 100 points and others score 0. Else, each player is awarded 1 (always) + 1 per supply center owned + 38 if topping the board + 14 for second place + 7 for third. In case of a tie for topping or for second/third, the points for those divided equally between those players.
- `"c_diplo_73"`: In case of solo, soloist scores 73 points and others score 0. Else, each player is awarded 1 (always) + 1 per supply center owned + 38 if topping the board + 14 for second place + 7 for third. In case of a tie for topping or for second/third, the points for those divided equally between those players.
- `"sum_of_centers"`: In case of solo, soloist scores 100% and others score 0%. Otherwise, score is divided proportional to owned centers.
- `"tribute"`: See https://windycityweasels.org/tribute-scoring-system/
- `"carnage"`: See https://www.playdiplomacy.com/forum/viewtopic.php?f=710&t=51339&start=10
- `"dixie"`: See https://www.dixiecon.com/tournament-rules

### Draws

If `draw_type` is `draw-votes-public` and `has_draw_votes` is `True`, then draw
votes are encoded as messages with the voting power as "sender" and "ALL" as
the "recipient", with the message body `<DRAW>`.
