# Corrupted Dialogue Data Teachers

A collection of teachers for training a discriminative classifier that can distinguish between nosiy/corrupted and real messages.
The data from each teacher is the game history concatenated with the dialogue (message that generator models train on).
There are two types of labels:
* *Real data label*, `[REAL]`, that goes with the data that directly comes from the training or validation data source: without any changes.
* *Corrupted data labels* that depend on `--corrupted-type-label`: if this label is set to `False` (default) then all the corrupted data (regadless of type of data corruption and noising applied to them) have `[CORRUPTED]` label.
Otherwise, the label shows the type of data corruption applied.
For the actual label that each corruptor class uses read the entry about that teacher below.

There are two set of teachers:

1. *Training teachers*: used for training the discriminator. Their outputs are randomized---may change between runs.
2. *Validation teachers*: they read data from a pre-generated set of data that are previously corrupted by training teachers. Their output is not randomized.

## Training Teachers
You may use these teachers by directly instantising their correpsonding classes (listed below), or from `diplom` command on command line using flag `--task` or `-t` and passing in their registered name.

### BaseRealDialogueChunkTeacher
This is the teacher that provides real (uncorrupted data)
Registered task name for command line use: `base_real_dialogue_chunk`.

### ConversationParticipantNoiseDialogueChunkTeacher
* Registered task name for command line use: `conversation_participants_corrupted_dialogue_chunk`.
* Corrupted data label: `[INCORRECT_RECIPIENT]`

### EntityCorruptedNoiseDialogueChunkTeacher
* Registered task name for command line use: `entity_corrupted_dialogue_chunk`.
* Corrupted data label: `[CORRUPTED_ENTITY]`

### IncorrectPhaseCorruptedNoiseDialogueChunkTeacher
* Registered task name for command line use: `incorrect_phase_message_corrupted_dialogue_chunk`.
* Corrupted data label: `[INCORRECT_PHASE]`

### IncorrectGameCorruptedNoiseDialogueChunkTeacher
* Registered task name for command line use: `incorrect_game_message_corrupted_dialogue_chunk`.
* Corrupted data label: `[INCORRECT_GAME]`

### RepeatMessageNoiseDialogueChunkTeacher
* Registered task name for command line use: `repeat_message_corrupted_dialogue_chunk`.
* Corrupted data label: `[REAPEATED_MESSAGE]`

## Validation Teachers

There are two teachers here: one for corrupted validation data and the other for real (uncorrupted).

### Corrupted validation data
`CorruptedMessagesValidTeacher` that is available with `diplom` command via `--task corrupted_dialogue_validation_teacher`.
The type of corrupted data to load may be set by `--valid-datatypes` flag. The available values are as follows (check their correspondence with training teachers):
* incorrect-recipient
* corrupted-entity
* incorrect-phase
* incorrect-game
* repeated-message
* tests (only used for unittests)

Note that validation teachers use `--corrupted-type-label` similar to the training teacher, and generate the same labels as their corresponding training teacher
(notice name similarities for correspondencce between training and validation teachers).

Use example

```
diplom dd -t corrupted_dialogue_validation_teacher \
--valid-datatypes=incorrect_game,repeated_message -dt train
```

### Corrupted validation data
`RealMessagesValidTeacher` is available by `diplom` command `--task real_dialogue_validation_teacher`.

### Validation data generation

> **WARNING**: Remember we create the valid data dumps to avoid random outcomes between validation runs.
Be cautious about creating a new validation data dump that replaces the current ones.

To generate a new dump of validation from corruptor teachers one may use `repeat_label` model from ParlAI:
```
diplom eval_model   \
--task <registered name of the task you want to use>    -dt valid:stream           \
-m "repeat_label"      --world-logs <path to save the data dump>
```
