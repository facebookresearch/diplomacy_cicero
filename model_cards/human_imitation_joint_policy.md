# human_imitation_joint_policy.ckpt model card

## Overview

This model was the anchor policy used during Cicero's RL to ensure Cicero stayed compatible with human conventions. It uses a transformer encoder on a representation of the gamestate with an LSTM decoder to autogressively decode the action to be predicted. It can decode actions for one or two, or all players at a time, the latter cases allowing the model to autoregressively predict corelations between players' actions due to unobserved private dialogue.

This model was developed by Meta AI. It was trained in April 2022.

### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).

### Architecture details

This model has roughly 8 million trainable parameters. See [here](https://www.science.org/doi/10.1126/science.ade9097) for details on the architecture and input and output encoding.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It may only implicitly predict the effects that the presence of dialogue may cause probabilitistically in expectation, by directly modeling the final action distribution of players.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097).

## Privacy

This model does not observe any personally identifiable information. It was trained to predict actions and/or values in Diplomacy games on an anonymized dataset of such games.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `batch_size`: `500`
 - `lr`: `0.002`
 - `lr_decay`: `0.99`
 - `clip_grad_norm`: `0.5`
 - `teacher_force`: `1.0`
 - `lstm_dropout`: `0.3`
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
 - `encoder`: `{'transformer': {'num_heads': 8, 'ff_channels': 224, 'num_blocks': 10, 'dropout': 0.3, 'activation': 'gelu'}}`
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
</details>



## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
