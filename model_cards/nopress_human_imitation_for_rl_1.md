# nopress_human_imitation_for_rl_1.ckpt model card

## Overview

This model is similar to [no_press_human_imitation_policy.ckpt](./no_press_human_imitation_policy.md) except it was independently trained on the same dataset, and that after training that its hyperparameters were manually edited to remove dropout and add new input features with randomly initialized weights to inform the model of the game year so that further training can allow the model to condition on year. This is used as the initial model for policy and value RL training in Diplodocus.

This model was developed by Meta AI. It was trained in November 2021.

See documentation on [no_press_human_imitation_policy.ckpt](./no_press_human_imitation_policy.md) for other details, including intended usage, datasets, privacy, etc.

### Architecture details

See Appendix E [here](https://arxiv.org/abs/2210.05492) for details on the architecture and input and output encoding.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `batch_size`: `500`
 - `lr`: `0.002`
 - `lr_decay`: `0.99`
 - `clip_grad_norm`: `0.5`
 - `teacher_force`: `1.0`
 - `lstm_dropout`: `0.0`
 - `num_epochs`: `400`
 - `value_loss_weight`: `0.7`
 - `value_decoder_init_scale`: `0.01`
 - `value_decoder_clip_grad_norm`: `0.5`
 - `lstm_layers`: `2`
 - `featurize_output`: `True`
 - `relfeat_output`: `True`
 - `featurize_prev_orders`: `True`
 - `dataset_params.only_with_min_final_score`: `0`
 - `dataset_params.exclude_n_holds`: `3`
 - `dataset_params.min_rating_percentile`: `0.5`
 - `dataset_params.min_total_games`: `5.0`
 - `value_softmax`: `True`
 - `encoder`: `{'transformer': {'num_heads': 8, 'ff_channels': 224, 'num_blocks': 10, 'dropout': 0.0, 'activation': 'gelu'}}`
 - `inter_emb_size`: `112`
 - `warmup_epochs`: `10`
 - `input_version`: `3`
 - `training_permute_powers`: `True`
 - `use_v2_dipnet`: `True`
 - `num_scoring_systems`: `2`
 - `launcher.slurm.num_gpus`: `32`
 - `use_year`: `True`
</details>

## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
