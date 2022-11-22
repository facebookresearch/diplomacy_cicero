# blueprint.pt model card

## Overview

This model was the policy proposal model used for some old baselines from past publications, trained to imitate human play via behavioral cloning in Diplomacy. It uses a graph convolution architecture that is obsoleted by more recent models with improved architectures that are both signficantly more accurate, similarly fast to evaluate, and have several times fewer parameters. This model was also jointly trained to predict final game scores.

This model was developed by Meta AI. It was trained mid-2020.

### Paper link

Jonathan Gray, Adam Lerer, Anton Bakhtin, and Noam Brown. Human-level performance in nopress diplomacy via equilibrium search. In International Conference on Learning Representations, 2020. https://arxiv.org/abs/2010.02923

### Architecture details

This model has roughly 24.5 million trainable parameters. See [here](https://arxiv.org/abs/2010.02923) for details on the architecture and input and output encoding.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It also decodes only one player's action at a time, predicting its probability independently from that of any other player.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://arxiv.org/abs/2010.02923).

## Privacy

This model does not observe any personally identifiable information. It was trained to predict actions and/or values in Diplomacy games on an anonymized dataset of such games.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

- `batch_size`: `1000`
- `lr`: `0.0010000000474974513`
- `lr_decay`: ` 0.9900000095367432`
- `clip_grad_norm`: `0.5`
- `val_set_pct`: `0.009999999776482582`
- `teacher_force`: `1.0`
- `lstm_dropout`: `0.10000000149011612`
- `encoder_dropout`: `0.20000000298023224`
- `learnable_A`: `false`
- `learnable_alignments`: `false`
- `avg_embedding`: `false`
- `num_encoder_blocks`: `8`
- `num_epochs`: `200`
- `value_loss_weight`: `0.8999999761581421`
- `value_decoder_init_scale`: `0.009999999776482582`
- `value_decoder_clip_grad_norm`: `1.0000000116860974e-07`
- `launcher`: `{'slurm' {'num_gpus': 8, 'single_task_per_node': true, 'partition': "learnfair", 'hours': 48, 'mem_per_gpu': 60, 'cpus_per_gpu': 10, 'volta': true}}`
</details>

## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
