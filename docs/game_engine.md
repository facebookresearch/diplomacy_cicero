# A Primer on the Game object

The game *environment* is defined by [fairdiplomacy.game.Game](../fairdiplomacy/game.py) object.

`game.get_state()` returns a dict containing the current board position. The most commonly accessed keys are:

`name`, returning the short-hand game phase:
```
>>> game.get_state()["name"]
'S1901M'
```

`units`, returning the locations of all units on the board:
```
>>> game.get_state()["units"]
{'AUSTRIA': ['A BUD', 'A VIE', 'F TRI'],
 'ENGLAND': ['F EDI', 'F LON', 'A LVP'],
 'FRANCE': ['F BRE', 'A MAR', 'A PAR'],
 'GERMANY': ['F KIE', 'A BER', 'A MUN'],
 'ITALY': ['F NAP', 'A ROM', 'A VEN'],
 'RUSSIA': ['A WAR', 'A MOS', 'F SEV', 'F STP/SC'],
 'TURKEY': ['F ANK', 'A CON', 'A SMY']}
```

`centers`, returning the supply centers controlled by each power:
```
>>> game.get_state()["centers"]
{'AUSTRIA': ['BUD', 'TRI', 'VIE'],
 'ENGLAND': ['EDI', 'LON', 'LVP'],
 'FRANCE': ['BRE', 'MAR', 'PAR'],
 'GERMANY': ['BER', 'KIE', 'MUN'],
 'ITALY': ['NAP', 'ROM', 'VEN'],
 'RUSSIA': ['MOS', 'SEV', 'STP', 'WAR'],
 'TURKEY': ['ANK', 'CON', 'SMY']}
```

`game.order_history` is a SortedDict of {short phase name => {power => [orders]}}
```
>>> game.order_history
{'S1901M': {'AUSTRIA': ['A VIE - GAL', 'F TRI H', 'A BUD - RUM'],
            'ENGLAND': ['F EDI - NTH', 'A LVP - YOR', 'F LON - ENG'],
            'FRANCE': ['F BRE - MAO', 'A PAR - BUR', 'A MAR S A PAR - BUR'],
            'GERMANY': ['F KIE - HOL', 'A BER - KIE', 'A MUN - BUR'],
            'ITALY': ['A VEN - PIE', 'A ROM - VEN', 'F NAP - ION'],
            'RUSSIA': ['A MOS - UKR',
                       'F STP/SC - BOT',
                       'A WAR - GAL',
                       'F SEV - BLA'],
            'TURKEY': ['A SMY - ARM', 'F ANK - BLA', 'A CON - BUL']},
 'F1901M': {'AUSTRIA': ['A VIE - GAL', 'F TRI H', 'A RUM S A ARM - SEV'],
 ...
```

`game.get_all_possible_orders()` returns a dict from location -> list of possible orders, e.g.
```
>>> game.get_all_possible_orders()
{'ADR': [],
 'AEG': [],
 'ALB': [],
 'ANK': ['F ANK S F SEV - ARM',
         'F ANK H',
         'F ANK S A CON',
         'F ANK - ARM',
         'F ANK S A SMY - CON',
         'F ANK S F SEV - BLA',
         'F ANK S A SMY - ARM',
         'F ANK - BLA',
         'F ANK - CON'],
  ...
```

`game.get_orderable_locations()` returns a map from power -> list of locations that need an order:
```
>>> game.get_orderable_locations()
{'AUSTRIA': ['BUD', 'TRI', 'VIE'],
 'ENGLAND': ['EDI', 'LON', 'LVP'],
 'FRANCE': ['BRE', 'MAR', 'PAR'],
 'GERMANY': ['BER', 'KIE', 'MUN'],
 'ITALY': ['NAP', 'ROM', 'VEN'],
 'RUSSIA': ['MOS', 'SEV', 'STP', 'WAR'],
 'TURKEY': ['ANK', 'CON', 'SMY']}
```

`game.set_orders(power, orders_list)` sets orders for that power's units, e.g.
```
>>> game.set_orders("TURKEY", ["F ANK - BLA", "A CON H"])
```

`game.process()` processes the orders that have been set, and moves to the next phase

