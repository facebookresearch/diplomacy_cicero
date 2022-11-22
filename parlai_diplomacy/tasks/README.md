# ParlAI Diplomacy Teachers

A teacher is an object which converts games in our diplomacy dataset to input/output example sequences that can be used to train a model. Different teachers produce outputs that can be used to train classification models, dialogue models, or any other model in our repo.

For example the [`message_history_orderhistorysincelastmovementphase_shortstate_allorder_chunk`](order/all_orders_agents.py) teacher is used to train a model which produces orders for all powers, based on the current messages, order history, and game state. By running:

```
diplom dd -t message_history_orderhistorysincelastmovementphase_shortstate_allorder_chunk --task-version 3
```

We can see an example input sequence:

```
S1901M
RUSSIA -> ALL: <ANONYMIZED MSG>
ENGLAND -> RUSSIA: <ANONYMIZED MSG>
FRANCE -> ENGLAND: <ANONYMIZED MSG>
GERMANY -> ENGLAND: <ANONYMIZED MSG>
RUSSIA -> ENGLAND: <ANONYMIZED MSG>
GERMANY -> ALL: <ANONYMIZED MSG>
ENGLAND -> FRANCE: <ANONYMIZED MSG>

units: AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, F EDI, F LON; FRANCE: A MAR, A PAR, F BRE; GERMANY: A BER, A MUN, F KIE; ITALY: A ROM, A VEN, F NAP; RUSSIA: A MOS, A WAR, F SEV, F STP/SC; TURKEY: A CON, A SMY, F ANK
S1901M ENGLAND:
```

And a corresponding output sequence:
```
FRANCE: A MAR S A PAR BUR; A PAR BUR; F BRE MAO
ITALY: A ROM APU; A VEN H; F NAP ION
GERMANY: A BER KIE; A MUN BUR; F KIE DEN
AUSTRIA: A BUD RUM; A VIE GAL; F TRI H
TURKEY: A CON BUL; A SMY CON; F ANK BLA
RUSSIA: A MOS SEV; A WAR UKR; F SEV RUM; F STP/SC BOT
ENGLAND: A LVP YOR; F EDI NTH; F LON ENG
```

which can be used to train a text-to-text transformer model.


## Where are the teachers actually implemented?

Astute observers will notice that the teacher classes, like the [`MessageHistoryOrderHistorySinceLastMovementPhaseShortStateAllOrderChunkTeacher`](order/all_orders_agents.py) class which implements the teacher described above, have an empty `pass` implementation. So where is the code?

Answer: The `register_teacher` annotation defines a task name which is parsed into pieces that each correspond to a part of the input or output sequence. For example, the string `allorder` in the task is parsed by [parlai_diplomacy/utils/game2seq/factory.py](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/utils/game2seq/factory.py#35) to return a formatter object which produces the output sequence. [The parent class](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/utils/game2seq/order_prediction.py#L58) parses strings like `message_history` to determine that the input sequence should contain the message history.


## What model types / output formats are implemented?

As seen in [parlai_diplomacy/utils/game2seq/factory.py](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/utils/game2seq/factory.py), we currently implement the following teacher types:

- Orders models:
   - `order`: teachers which output orders for a single power for the current phase
   - `orderrollout`: teachers which output orders for a single power for the current phase and future phases until the next movement phase (i.e. the output will always terminate with orders for an M-phase)
   - `allorder`: teachers which output orders for all powers for the current phase
   - `allorderindependent`: teachers which output orders for our agent and one other target/recipient power
   - `allorderrollout`: teachers which output orders for all powers, rolled out to the next movement phase
   - `allorderindependentrollout`: teachers which output orders for our agent and one other target/recipient power, rolled out to the next movement phase
   - `plausiblepseudoorder`: teachers which output the *annotated* pseudo orders corresponding to a given message, which corresponds to the pseudo orders used to condition the dialogue model for predicting that message
- Dialogue models:
   - `dialogue`: teachers which output a natural language message
- Classifiers:
   - `dialoguediscriminator`: teachers which output a classification label predicting whether a message is nonsensical (and should be filtered / not sent)
   - `sleepclassifier`: teachers which output a classification label predicting how long an agent should wait before sending a message
   - `sleepsix`: teachers which output a classification label predicting how long an agent should wait before sending a message *to a particular recipient*
   - `recipientclassifier`: teachers which output a classification label predicting who the agent should speak to next
   - `drawclassifier`: teachers which output whether or not to submit a draw at a given time
   - `liedetector`: teachers which output a classification label predicting whether a power is currently lying


## Teacher Flags

The teacher name alone does not determine the input/output sequence formatting; there are a number of flags which apply modifications. For example, the input sequence for the task above terminates with the prompt:

```
... S1901M ENGLAND:
```

which indicates that the agent is playing as England and that the current phase is S1901M. However the same task with the `--include-game-info True` flag changes the prompt:

```
... S1901M ENGLAND ANON 5min WTA:
```

Here the prompt has additional information appended:
- `ANON` means the game is anonymous on webdip
- `5min` means the phases have a 5 minute time limit
- `WTA` means the game uses "winner takes all" scoring, vs. `SOS` "sum of squares" scoring

These flags are specified in the teacher's `add_cmdline_args` method, and many of them are defined for all tasks in the `BaseDiplomacyTeacher` in [parlai_diplomacy/tasks/base_diplomacy_agent.py](base_diplomacy_agent.py)