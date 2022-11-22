# human_sl_value_function.ckpt model card

## Overview

This model is a value model trained based on short rollouts of the no_press_human_imitation_policy via supervised learning to predict the expected final score of the game based on those rollouts. It uses a transformer encoder on a representation of the gamestate followed by a small value head to predict the score for all 7 players.

This model was developed by Meta AI. It was trained May 2022.

### Paper link

See the improved value modeling method in Jacob et al., Modeling strong and human-like gameplay with kl-regularized
search. ICML 2022. https://arxiv.org/abs/2112.07544

### Architecture details

This model has roughly 3 million trainable parameters. See [here](https://arxiv.org/abs/2010.02923) for details on the architecture and input and output encoding.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue.

## Datasets used

This model was trained using rollouts from another model, via method in described in https://arxiv.org/abs/2112.07544, on game positions sampled from the webDiplomacy dataset, described in the [paper](https://arxiv.org/abs/2210.05492).

## Privacy

This model does not observe any personally identifiable information. It was trained to predict values in Diplomacy games using a method that sampled gamestates from an anonymized dataset of such games with another model trained on those games.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `lstm_dropout`: `0.0`
 - `value_loss_weight`: `0.7`
 - `value_decoder_init_scale`: `0.01`
 - `value_decoder_clip_grad_norm`: `0.5`
 - `lstm_layers`: `2`
 - `featurize_output`: `True`
 - `relfeat_output`: `True`
 - `featurize_prev_orders`: `True`
 - `value_softmax`: `True`
 - `encoder`: `{'transformer': {'num_heads': 8, 'ff_channels': 224, 'num_blocks': 10, 'dropout': 0.0, 'activation': 'gelu'}}`
 - `inter_emb_size`: `112`
 - `input_version`: `3`
 - `training_permute_powers`: `True`
 - `use_v2_dipnet`: `True`
 - `num_scoring_systems`: `2`
 - `value_decoder_activation`: `gelu`
 - `value_decoder_use_weighted_pool`: `True`
 - `has_policy`: `False`
 - `has_value`: `True`
 - `use_year`: `True`
 - `value_dropout`: `0.0`
- `critic_weight`: `1.0`
- `optimizer.grad_clip`: `0.5`
- `optimizer.warmup_epochs`: `100`
- `optimizer.adam.lr`: `1e-05`
- `discounting`: `1.0`
- `search_rollout.agent.searchbot.n_rollouts`: `40`
- `search_rollout.agent.searchbot.use_final_iter`: `False`
- `search_rollout.agent.searchbot.plausible_orders_cfg.n_plausible_orders`: `50`
- `search_rollout.agent.searchbot.plausible_orders_cfg.max_actions_units_ratio`: `3.5`
- `search_rollout.agent.searchbot.plausible_orders_cfg.req_size`: `1024`
- `search_rollout.agent.searchbot.half_precision`: `True`
- `search_rollout.num_workers_per_gpu`: `8`
- `search_rollout.chunk_length`: `128`
- `search_rollout.batch_size`: `8`
- `search_rollout.extra_params.explore_eps`: `0.0`
- `search_rollout.extra_params.independent_explore`: `True`
- `search_rollout.extra_params.use_trained_policy`: `False`
- `search_rollout.extra_params.explore_s1901m_eps`: `0.0`
- `search_rollout.extra_params.explore_f1901m_eps`: `0.0`
- `search_rollout.extra_params.run_do_prob`: `0.0`
- `search_rollout.extra_params.use_ev_targets`: `False`
- `search_rollout.extra_params.use_trained_value`: `False`
- `search_rollout.extra_params.always_play_blueprint`: `{'temperature': 0.75, 'top_p': 0.949999988079071}`
- `search_rollout.extra_params.sample_game_json_phases`: `True`
- `search_rollout.extra_params.min_max_episode_movement_phases`: `2`
- `search_rollout.extra_params.max_max_episode_movement_phases`: `4`
- `search_rollout.extra_params.max_training_episode_length`: `1`
- `search_rollout.buffer.capacity`: `10000`
- `search_rollout.enforce_train_gen_ratio`: `2.0`
- `search_rollout.draw_on_stalemate_years`: `3`
- `search_policy_weight`: `0.0`
- `bootstrap_offline_targets`: `True`
- `num_train_gpus`: `4`
- `use_distributed_data_parallel`: `True`
- `launcher.slurm.num_gpus`: `64`
</details>

## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
