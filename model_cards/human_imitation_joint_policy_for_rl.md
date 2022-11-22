# human_imitation_joint_policy_for_rl.ckpt model card

## Overview

This model is the same as [human_imitation_joint_policy.ckpt](./human_imitation_joint_policy.md) except that after training its hyperparameters were manually edited to remove dropout and add new input features with randomly initialized weights to inform the model of the game year so that further training can allow the model to condition on year. This is used as the initial model for policy training in Cicero.

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
 - `value_loss_weight`: `0.5`
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
 - `all_powers`: `True`
 - `warmup_epochs`: `10`
 - `input_version`: `3`
 - `training_permute_powers`: `True`
 - `use_v2_dipnet`: `True`
 - `num_scoring_systems`: `2`
 - `value_decoder_activation`: `gelu`
 - `value_decoder_use_weighted_pool`: `True`
 - `all_powers_add_single_chances`: `4.0`
 - `all_powers_add_double_chances`: `4.0`
 - `single_power_conditioning_prob`: `0.5`
 - `with_order_conditioning`: `True`
 - `launcher.slurm.num_gpus`: `32`
 - `use_year`: `True`
 - `value_dropout`: `0.0`
</details>


## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
