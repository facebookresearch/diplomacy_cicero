# no_press_human_imitation_policy.ckpt model card

## Overview

This model was the anchor policy used during Diplodocus's RL and during actual play to ensure Diplodocus stayed compatible with human conventions. It uses a transformer encoder on a representation of the gamestate with an LSTM decoder to autogressively decode the action to be predicted. It was also jointly trained to predict the expected value for all players in the game.

This model was developed by Meta AI. It was trained in November 2021.

### Paper link

Find more details in our paper [here](https://arxiv.org/abs/2210.05492).

### Architecture details

This model has roughly 8 million trainable parameters. See Appendix F [here](https://arxiv.org/abs/2210.05492) for details on the architecture and input and output encoding. Note that this model however instead uses a slightly improved value head from figure S7 [here](https://www.science.org/doi/10.1126/science.ade9097).

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It also decodes only one player's action at a time, predicting its probability independently from that of any other player.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://arxiv.org/abs/2210.05492).

## Privacy

This model does not observe any personally identifiable information. It was trained to predict actions and/or values in Diplomacy games on an anonymized dataset of such games.

## Other Related Paper(s)

Jonathan Gray, Adam Lerer, Anton Bakhtin, and Noam Brown. Human-level performance in nopress diplomacy via equilibrium search. In International Conference on Learning Representations, 2020. https://arxiv.org/abs/2010.02923

Anton Bakhtin, David Wu, Adam Lerer, and Noam Brown. No-press diplomacy from scratch. In
Thirty-Fifth Conference on Neural Information Processing Systems, 2021. https://arxiv.org/abs/2110.02924

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
 - `encoder`: `{'transformer': {'num_heads': 8, 'ff_channels': 224, 'num_blocks': 10, 'dropout': 0.3, 'activation': 'gelu'}}`
 - `inter_emb_size`: `112`
 - `warmup_epochs`: `10`
 - `input_version`: `3`
 - `training_permute_powers`: `True`
 - `use_v2_dipnet`: `True`
 - `num_scoring_systems`: `2`
 - `value_decoder_activation`: `gelu`
 - `value_decoder_use_weighted_pool`: `True`
 - `launcher.slurm.num_gpus`: `32`
</details>



## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
