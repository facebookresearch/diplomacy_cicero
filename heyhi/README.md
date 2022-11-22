# HeyHi

HeyHi is a libary for managing configuration and launching jobs.
HeyHi is the thing that takes arguments passed to `run.py` and at the end launches something.The following steps are handled by HeyHi:

  1. Construct the config (hierarchical set of flags) from the base config stored on disk and any changes to this config from the command line
  2. Understand what kind function this config should be passed to (e.g., train or h2h eval)
  3. Determine what's the output folder should be if not explicitly specified by the user
  4. Define the execution context - locally or on slurm
  5. Set working dir to the output folder, launch the job, and pass the config to the function


The launching module is one of the components of HeyHi. The general structure looks like this:

  * A centralized schema storage of all possible tasks and their flags
  * A logic to combine on-the-disk config files with overrides and verify against schema
  * Python representation of the config object that provides typing support, immutability, etc
  * The `run.py` logic that combines everything together.


Below we go into details of different steps.

## run.py

The signatute of the `run.py` looks likes this (some flags are skipped, see `--help` for full list):

```bash
python run.py -c CFG [--adhoc]
              [--mode {gentle_start,restart}]
              [--out OUT]
              [--print] [--print-flat] [--print-flat-all]
              [overrides]
```

You can see that `run.py` has one required argument: the path to the base config, e.g.,

```bash
python run.py --cfg conf/c02_sup_train/sl.prototxt
```

This will read the config from the file and pass it to `run.train(cfg)` function in run.py. This job will run locally (default). The output of the job will go to `~/results/diplomacy/p/c02_sup_train/sl/default_d7517139`. The name of output folder is a function of the input parameters, therefore if you run the same command twice, the output will be the same.
By default, HeyHi will not run job again if the output folder exists.
This is useful, e.g., if you have bash script that have several `run.py` commands to train different models. In that case, you can just add more commands to this file, re-run it and only new jobs will be launched.

Sometimes that's not the behavior one expects. There are few ways to change this:
  * With `--adhoc` flag, HeyHi will add current date to the output folder so that it is unuque.
  * If `--model restart` set and the output folder exists, HeyHi will kill the corresponding slurm job (if any), wipe the folder, and start a new.
  * Finally, one can specicy output folder manually using `--out` flag.

The rule of thumb is that for all debugging or test runs always add `--adhoc`.

Flags `[--print] [--print-flat] [--print-flat-all]` allow to print the config to be executed after all command line overrides are applied. The `--print` flag will show the config in heirarchical way so that it's easy examing. `--print-flat` and `--print-flat-all` will print the config in flag way, i.e., in a format of overrides, e.g.,:

```bash
$ python run.py --cfg conf/c02_sup_train/sl.prototxt --print-flat
avg_embedding=False
batch_size=1000
checkpoint=./checkpoint.pth
clip_grad_norm=0.5
dataset_params.debug_only_opening_phase=False
```

Finally, `overrides` is a list of things we want to change in the base config, e.g.,
```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt batch_size=200 --dataset_params.value_decay_alpha=1.0
```

Note that it's optional to use "--" in front of overrides. It's better not to use it so that it's easier to understand what is a flag to be handled by HeyHi and what is an override that will change the config for the user's function to use.

On the conceptual level, overrides is a set of key-value pairs that combines with the base config to create the final config.


## Configs

Any configuration system consists of 4 things:
  * Schema description (what flags exists, e.g., add_flag in argparse)
  * Input representation of the config instance (e.g., a `sys.argv` list for argparse)
  * Logic to parse the latter using the former (e.g., `argparse.parse_args`)
  * A representation of the config instance in memory (e.g., `argparse.Namespace`)

In HeyHi we use [protobuf messages](https://developers.google.com/protocol-buffers/docs/proto#simple) to define this schema, as it allows to defines hierarchical structures and some polymorphism.

### Brief proto intro

In general, protobuf is a language to define structured data. E.g., here's how one can defines arguments for an optimizer:

```proto
message Optimization {
  // Required. Learning rate.
  optional float lr = 1;
  // Optional (but highly recommended). Gradient clipping.
  optional float grad_clip = 2;
  oneof {
    int32 warmup_epochs = 3;
    int32 warmup_hours = 4;
  }
}
```

A message could be see as analogue of `struct` or `dataclass.dataclass`. It simply says that an object of type `Optimization` has 4 fields. Keyword `optional` and ` = XXX` refers to additional features of protobuf that we don't use, so you should mentally ignore them when reading a schema.

One the hand `oneof` is one of the key features that allows to define polymorphism within configs. Here we say that for warmup we can use either epochs or hours, but not both.

### Tasks

All schemas are stored in `*.proto` files in [conf/](../conf/) folder.
All top-level configs, i.e., configs you feed to `run.py` are instances of `MetaCfg` as defined in [conf/conf.proto](../conf/conf.proto).
Here's how `MetaCfg` looks like.

```proto
message MetaCfg {
  repeated Include includes = 1;
  oneof task {
    CompareAgentsTask compare_agents = 101;
    TrainTask train = 102;
    // ...
  }
}
```

In a nutshell `MetaCfg` says that a user has to specify one of the messages for specific task and maybe a list of includes. Include is simply a message with two fields: `path` and `mount`. Includes allow to "include" one config as a part of another.


Here's an example of an instance of MetaCfg with include:
```
includes { path: "launcher/slurm_8gpus.prototxt"; mount: "launcher" }
train {
    batch_size: 1000;
    lr: 0.001;
    lr_decay: 0.99;
    clip_grad_norm: 0.5;
    checkpoint: "./checkpoint.pth";
    // ...
}
```

This says:
  * This a `train` config, i.e., we select `train` from the `task` oneof in the `MetaCfg`
  * We want to read file [launcher/slurm_8gpus.prototxt](../conf/common/launcher/slurm_8gpus.prototxt) and merge it's content into the `launcher` field of this config.


  If you run `python run.py --cfg <this_cfg file>`, this config will be composed to produce a final `TrainTask` message to pass to python code:

   * HeyHi will read the config above and parse it as `MetaCfg`.
   * HeyHi will look into definition of TrainTask in conf.proto to see that `launcher` is a submessage of type `Launcher`.
   * HeyHi will read `launcher/slurm_8gpus.prototxt` and parse it as a Launcher message
   * HeyHi will merge the content of the include into the main config
   * HeyHi will pass the config object to `run.py -> train(cfg)` function.


Specific configs are stored in [conf/](../conf/) folder.
E.g., all config to train agents are in `conf/c02_sup_train/`.

[conf/common](../conf/common) containing config chunks that could be included into other configs. Currently, there 2 kind of chunks: launchers (as in example above) and agents.

Here's an example of agent includes in our [h2h eval task](../conf/c01_ag_cmp/cmp.prototxt):
```
includes {path: "agents/searchbot"; mount: "agent_one";}
includes {path: "agents/base_strategy_model"; mount: "agent_six";}
compare_agents {
  power_one: AUSTRIA
  seed: 0
  out: "output.json"
}
```

The behavior is identical to the case of launcher includes. 1) We detect the types of `agent_one` and `agent_six` from the schema. 2) We read the files and parse them according to the subtypes. 3) We merge the content into the main config.

If you are not sure how composition will work, there are tools to help you:
```
python run.py --cfg conf/c01_ag_cmp/cmp.prototxt --print
bin/pp.py conf/c01_ag_cmp/cmp.prototxt
```

The two commands are almost identical. One is easier to use if you already started doing `run.py`, and the other one is shorter :p

Final note about imports. Imports is a powerful tool to reduce duplication, but it's also a dangerous tool as can create long complex dependencies and make config harder to read. Due to that reason we allow to specify `includes` fields only for 2 types of configs: `MetaCfg` and `Agent`.

### Command line overrides

Now you should know where to find configs to train or eval agents (conf/*/*.prototxt), where to find all the flags (conf/*.proto), and how to launch an baseline.

Now we talk about the most important function - command line overrides.
Configs allow to capture a "good state of the world". A new baseline to train a model or a new combination of models for an agent. But between baselines we want to be able to modify minor parts of the config to test different changes.

We have to types of overrides:
  * Scalar overrides, e.g., for number of epochs or sampling temperature. Format `<key>=<value>`.
  * Include overrides, e.g., when we want to include some agent into eval or a dialogue model into an agent. Format: `I<mount_point>=<config_path>`.

Here's an example of a scalar redefine for base_strategy_model training config c02_sup_train/sl_20200901.prototxt:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl_20200901.prototxt \
    batch_size=10
    encoder.transformer.num_heads=9
```

Redefining batch size is pretty straightforward. Setting num heads to an odd number is questionable, but otherwise gives an idea of how overrides handle hierarchical structure.

It also possible to reset some flag to default value by setting this to `NULL`:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl_20200901.prototxt \
    batch_size=NULL \
    encoder.transformer=NULL
```

The first overrides sets batch_size to its default value (that is 0, so not good idea in this case). This second overrides remove `encoder` stanza altogether. If you open the schema for the encoder you can see that it is defines as `oneof`:
```
  oneof encoder { Transformer transformer = 1; }
```

Therefore, by setting it to `NULL` we reset the choice between options.

Here's an example of import include:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl_20200901.prototxt \
    Ilauncher=slurm_8gpus
```

The logic is identical to what would happen for include inside a config itself (see example above in this doc). Interesting thing is that we can mix scalar and include overrides:

```
python run.py --adhoc --cfg conf/c02_sup_train/sl_20200901.prototxt \
    Ilauncher=slurm_8gpus \
    launcher.slurm.partition=Diplomacy
```

This will first merge the include from the file, and then change the name of the partition.
You may wonder how `slurm_8gpus` becomes `conf/common/slurm_8gpus.prototxt`.
We a standard logic with search paths. HeyHi will try to find the include relative to the folder of the base config, i.e., `conf/c02_sup_train/` and then check `conf/common`. If you do something special, you can use absolute path for include.

Finally, in real world you may have includes in the config, in overrides, and scalar overrides. In these cases it's useful to know the order of composition:

  1. Read the includes within the file and merge into an **empty** version of the task
  2. Merge the base config
  3. Merge command line include-overrides
  4. Apply scalar overrides.



### Configs in python

If you open some diplomacy code, chances are there is come `cfg` object somewhere there. The code for configs for `conf/conf.proto` lives in `conf/conf_cfgs.py` (these files are generated on `make protos`). The code in this files is cryptic, but there are interface files, ``conf/conf_cfgs.pyi` that tells how the config objects look like. E.g.,:

```python
class TrainTask(google.protobuf.message.Message):
    def is_frozen(self) -> builtins.bool: ...
    def to_editable(self) -> 'conf.conf_pb2.TrainTask': ...
    def to_frozen(self) -> TrainTask: ...
    def to_dict(self, *, with_defaults: builtins.bool = False, with_all: builtins.bool = False) -> Dict[str,Any]: ...
    def to_str_with_defaults(self) -> str: ...
    # ...
    # No Press dataset params
    @property
    def dataset_params(self) -> global___NoPressDatasetParams: ...
    # Batch size per GPU.
    @property
    def batch_size(self) -> Optional[builtins.int]: ...
    # Learning rate.
    @property
    def lr(self) -> Optional[builtins.float]: ...
```

In a nutshell, the config object will look like a `dataclass`.
To debug the configs in python one could use `heyhi.load_config` function.



## FAQ

### Nothing works
If you get error about failed import `conf.conf_pb2` or about missing fields in the config, run:
```
make protos
```

### If I launch a job on slurm and change the code, will my job use the new code

No, by default HeyHi will create a copy of the repo at the moment of the launch. Use `--checkpoint=""` to disable checkpointing.

### Why things run sometimes on cluster and sometimes locally

The `Launcher` messages is oneof between `local` and `slurm`. If `slurm` is not set, then jobs is launched locally.


### How do I do `--help`
Go to `conf/conf.proto` and you'll see all flags with docs.

### Redefine output folder
Use `--out=<outdir>`.

### Adding new flags
Just add them into the proto. The field id (number after "=") doesn't matter as we don't use binary format. Just increment the last one. Don't forget to run `make` after this.

## Advanced usage

### HeyHi API

Check `heyhi/__init__.py` for the list of all functions that are considered public. They all have solid docstrings.