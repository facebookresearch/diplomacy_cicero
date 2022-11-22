# ParlAI Diplomacy

Directory for training and evaluation of all things language-related in Diplomacy. Model training and evaluating is largely done in the
[ParlAI framework](https://github.com/facebookresearch/ParlAI).

Directory structure:
- **agents**: custom ParlAI agents (models)
- **metrics**: custom metrics, e.g. related to order prediction accuracy
- **scripts**: scripts for training, evaluation, etc.
- **tasks**: Diplomacy tasks and teachers for training and evaluation
- **utils**: all utilities for training and evaluation; in particular, utilities in game2seq govern all conversions from game/structured objects into strings and vice versa
- **wrappers**: wrappers for ParlAI agents which define APIs that allow the agents to interact with the game environment


For users who do not have access to training data, section 3 -- deploying models in the game environment -- will be the most relevant documentation.

## Table of Contents
1. [**TRAINING** Training in the ParlAI framework](#training-in-the-parlai-framework)
2. [**EVALUATING** Evaluating models on the dataset](#evaluating-models-on-thedataset)
3. [**DEPLOYING WRAPPERS:** Deploying ParlAI models in the Diplomacy Game Environment](#deploying-ParlAI-models-in-the-Diplomacy-Game-Environment)


## Training in the ParlAI framework


### Basic ParlAI commands
Please review the [ParlAI docs](https://parl.ai/docs/index.html) to learn basic familiarity with ParlAI. In particular, this [Colab Tutorial](https://colab.research.google.com/drive/1bRMvN0lGXaTF5fuTidgvlAl-Lb41F7AD#scrollTo=KtVz5dCUmFkN) may be useful.

### Diplom super command
All `parlai` commands are accessible in Diplomacy using the `diplom` super command. Using `diplom` allows us to register new tasks and agents inside `fairdiplomacy` so that they are visible to the `parlai` module. We can add new diplomacy specific shortcuts with this command as well. Read the next section for examples of using the super command to view data or train models.

### ParlAI data

**Viewing the data**

All tasks and teachers are located in [`parlai_diplomacy/tasks/`](https://github.com/facebookresearch/diplomacy_cicero/tree/main/parlai_diplomacy/tasks). The data can be viewed using the `diplom` super command. Here are some examples:

View a task for predicting orders given state information:
```
diplom dd -t state_order_chunk -dt train
```

View some validation examples for the task for for predicting dialogue given message history and state information:
```
diplom dd -t message_history_state_order_chunk -dt valid
```

The same as before, but with player rating information:
```
diplom dd -t message_history_state_order_chunk -dt valid --include-player-ratings True
```

> *NOTE:* You can view the list of available command line arguments for teacher formatting [here](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/tasks/base_diplomacy_agent.py) at the `BaseDiplomacyTeacher`'s `add_cmdline_args` function. Child classes may have their own list of arguments as well, such as for the [`BaseDialogueTeacher`](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/tasks/dialogue/base_agent.py).

**Creating a new task**

Again, all tasks and teachers are located in [`parlai_diplomacy/tasks/`](https://github.com/facebookresearch/diplomacy_cicero/tree/main/parlai_diplomacy/tasks). Notably, all teachers inherit from a singular Diplomacy base teacher, found [here](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/tasks/base_diplomacy_agent.py).

To create a new task which inherits from the base teacher, you willl have to register your agent with a task name that **directly corresponds to the input and output formats**. See `parlai_diplomacy/utils/game2seq/factory.py` for info about how the task name string maps to the input/output pieces.

The format must be defined using one of the format helpers in `parlai_diplomacy/utils/game2seq`.

See an example [here](https://github.com/facebookresearch/diplomacy_cicero/blob/main/parlai_diplomacy/tasks/order/single_order_agents.py) for the `StateOrderChunk` teacher. The `output_type` is `order` (as returned by `get_output_type` in `parlai_diplomacy/utils/game2seq/factory.py`) since the task teachers a model to generate an order for a given power. The input format type (as returned by `get_input_format` in `parlai_diplomacy/utils/game2seq/factory.py`) is `state`, as the model takes state information as input to predict orders. This format -- `state_order` is defined by the `OrderPredictionFormatter` in `parlai_diplomacy/utils/game2seq/order_prediction.py`. We define the format helpers here, so that when we deploy our models in a game environment, we can use the same format helper to convert a game object to a string that can be given as input to the Transformer model.


### Training models

#### **Training from the commandline**

Example training command for the dialogue model
```
python -u parlai_diplomacy/scripts/distributed_train.py --datapath <PATH TO PARLAI HOME DATA> --special-tok-lst $'[REDACTED],NON-ANON,HASDRAWS,Austria,England,Germany,AUSTRIA,ENGLAND,GERMANY,ALL-UNK,PRIVATE,NODRAWS,France,Russia,Turkey,FRANCE,RUSSIA,TURKEY,SPA/NC,STP/SC,BUL/SC,STP/NC,BUL/EC,SPA/SC,PUBLIC,Italy,ITALY,ANON,PPSC,VEN,ALB,KIE,BAR,NWG,TUS,EDI,GRE,PRU,BUD,HEL,IRI,SKA,GAL,TYS,RUM,NAP,SMY,LON,ADR,BOH,EAS,BEL,ANK,MAR,APU,TUN,PIE,SPA,HOL,SIL,MUN,YOR,LYO,ION,TYR,CON,WES,ENG,NAF,UKR,AEG,SER,ROM,WAR,BUR,VIA,VIE,LVP,GAS,BAL,BUL,BLA,TRI,ARM,SWE,RUH,NTH,NWY,BOT,DEN,NAO,WAL,BER,PIC,MOS,STP,BRE,PAR,SEV,MAO,SYR,FIN,LVN,CLY,POR,BAD,SOS,WTA,->' --load-from-checkpoint True --skip-input-validation True -t message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_dialogue_chunk --mark-bad-messages phase_repeats,offensive_language,redacted --task-version 3 --include-game-info True --include-draw-info True --include-centers-state True --include-draw-state True --hide-empty-draw-state False --single-view-pseudo-orders True --rollout-pseudo-orders True --rollout-except-movement False --add-sleep-times True --include_player_ratings True --include-sleep-messages False --filter-bad-messages-about-draws True --attention-dropout 0.0 --dropout 0.1 --relu-dropout 0.0 -m bart --init-model <PATH TO R2C2 MODEL WITH EXTENDED POSITIONAL EMBEDDINGS> --dict_file <PATH TO R2C2 DICT> --embedding-size 2048 --n-layers 22 --ffn-size 8192 --n-heads 32 --learn-positional-embeddings True --embeddings-scale True --n-segments 0 --variant prelayernorm --activation gelu --n-encoder-layers 22 --n-decoder-layers 22 --force-fp16-tokens True --dict-tokenizer gpt2 --n-positions 2048 --text-truncate 2048 --label-truncate 512 -lr 0.0001 --lr-scheduler linear --warmup-updates 8000 --warmup-rate 0.0001 --max-train-steps 100000 --update-freq 1 --fp16 True --fp16-impl safe --optimizer adam --gradient-clip 0.1 --skip-generation True -vp 100 -vmt ppl -vmm min -lstep 100 -vstep 1000 --save-after-valid True --checkpoint-activations True --ddp-backend zero2 --dynamic-batching None --num-workers 8 -bs 2 --model-file <WHERE TO SAVE THE TRAINED MODEL? --port 39968
```

## Evaluating models on the dataset

#### **Displaying model predictions on the training data**

You can display a model's predictions on a task (for debugging or evaluation purposes) using the `display_model` (shorthand: `dm`) script. For example:

```
diplom dm  --skip-generation False --inference topk -mf /checkpoint/fairdiplomacy/press_diplomacy/models/edinan/previousmovementorder/model -dt train:evalmode -t message_history_orderhistorysincelastmovementphase_pseudoorder_dialogue_chunk --include-player-ratings True
```

Note the `--skip-generation False` above: this is important as by default we skip generation during training for speed. Also note `-dt train:evalmode`: this is necessary to indicate to the model that we do not want it to train on these examples. Take care to make sure that any task flags are the same setting as your model was trained on.

#### **Evaluating models on the validation set**

Similarly, evaluating models can be achieved with the `eval_model` script (shorthand: `em`). Distributed evaluation -- across many GPUs -- is also possible. Please see `parlai_diplomacy/scripts/sweeps` for examples of evaluation sweeps.

### Creating new model architectures

If you want to create a new model architecture that is not available in public ParlAI, you can create and register a new agent in `parlai_diplomacy/agents`. This should only be done in rare cases, so please sync with the rest of the team before trying this. Again, public ParlAI docs will be most useful to you here.

#### **Model hyperparameters**

*To view the hyperparameters used to produce a model*, inspect the `model.opt` JSON file located in the same folder of the model file to see every single hyperparameter the model was trained with. E.g. the parameters used to generate `<model_dir>/model` can be found in `<model_dir>/model.opt`.



## Deploying ParlAI models in the Diplomacy Game Environment

So you've trained a ParlAI model. What now?

### The game environment
The diplomacy game environment is defined [here](https://github.com/facebookresearch/diplomacy_cicero/blob/main/fairdiplomacy/env.py).

Perhaps most relevant to ParlAI users, the **messaging protocol** is defined via the class `TimeBasedMessageRunner` in the same file. A **sleep classifier** predicts a time for the agent to sleep, the agent with the shortest sleep time is woken up and asked to send a message.

### Full press agent

The ParlAI full press agent -- which both speaks and takes actions in the game -- is defined [here](https://github.com/facebookresearch/diplomacy_cicero/blob/main/fairdiplomacy/agents/parlai_full_press_agent.py).

This agent has both an `order_handler` for executing orders and a `message_handler` for responding to dialogue. Each may have several components. For example, the message handler may consist of: (1) a generative dialogue model, (2) a sleep classifier for determining when to speak, (3) a recipient classifier, for determining who to speak to, (4) a pseudo-order generator, for generating orders for the dialogue model to condition its responses on, and (5) any number of other modules for testing the response generation, such as nonsense detectors. Each of these components is a *wrapper* for a ParlAI agent. Please see the next section to learn more about the wrappers.

### Wrappers for ParlAI agents

Wrappers for ParlAI agents define APIs so that we may pass a game object to the wrapper and get a structured object out of it. For example, the orders wrapper takes a game object and using the specific model information to convert that to a string ingestible by the model. It then takes the model's output string and converts it to a structured set of orders, to be returned to the `order_handler`. The wrapper may also contain metadata for the model, like which player rating it should use.

Wrappers for ParlAI model types (orders, dialogue, etc.) are defined in `parlai_diplomacy/wrappers`. Which wrapper should be instantiated for a given model file and set of arguments is determined in `parlai_diplomacy/wrappers/factory.py`. You can see common APIs defined by the base wrapper in `parlai_diplomacy/wrappers/base_wrapper.py`. The formatter loaded should be determined by the task name. Any special formatting code should be contained in this formatter, so that it is shared with the training task.

### Head to head matches and configs

Assuming you have a well-defined wrapper for your ParlAI model, you can now use it as part of a full press Diplomacy agent in the game environment.

In order to do so, you must define a config for your full press agent [here](https://github.com/facebookresearch/diplomacy_cicero/tree/main/conf/common/agents). There are separate config folders for order and message handler setups. The README at this location additionally gives some examples for running head to head matches with ParlAI configs.

More info about using heyhi can be found [here](https://github.com/facebookresearch/diplomacy_cicero/tree/main/heyhi).
