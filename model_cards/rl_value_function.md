# rl_value_function.ckpt model card

## Overview

This model central to Cicero's planning in that it is the source of all value estimates for all gamestates, which Cicero uses to judge which outcomes would be good or bad for different players and by how much. It was trained jointly with [rl_search_orders.ckpt](./rl_search_orders.md) via Deep Nash Value Iteration using Correlated Best Response as the search algorithm. The training code uses [human_sl_value_function.ckpt](./human_sl_value_function.md) as the initialization for this model.

This model was developed by Meta AI. It was trained in August 2022.

### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).

### Architecture details

This model has roughly 3 million trainable parameters. See [here](https://www.science.org/doi/10.1126/science.ade9097) for details on the architecture and input and output encoding and training method, particularly Supplementary Materials section E.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It may only implicitly predict the effects that the presence of dialogue may cause probabilitistically in expectation, by directly modeling the final action distribution of players.

## Datasets used

This model was trained via RL from baseline models that were trained using the webDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097).

## Privacy

This model does not observe any personally identifiable information.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

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
- `optimizer.grad_clip`: `0.5`
- `optimizer.warmup_epochs`: `100`
- `optimizer.adam.lr`: `0.0001`
- `search_rollout.agent.best_agent.num_br_samples`: `100`
- `search_rollout.agent.best_agent.qre_lambda`: `0.03`
- `search_rollout.agent.best_agent.anchor_joint_policy_model_path`: `models/human_imitation_joint_policy.ckpt`
- `model_path`: `models/human_imitation_joint_policy_for_rl.ckpt`
- `value_model_path`: `models/human_sl_value_function.ckpt`
- `search_rollout.agent.best_agent.plausible_orders_cfg.n_plausible_orders`: `10`
- `search_rollout.agent.best_agent.plausible_orders_cfg.max_actions_units_ratio`: `6.0`
- `search_rollout.agent.best_agent.plausible_orders_cfg.req_size`: `512`
- `search_rollout.agent.best_agent.half_precision`: `True`
- `search_rollout.num_workers_per_gpu`: `4`
- `search_rollout.chunk_length`: `128`
- `search_rollout.batch_size`: `8`
- `search_rollout.extra_params.explore_eps`: `0.0`
- `search_rollout.extra_params.independent_explore`: `True`
- `search_rollout.extra_params.use_trained_policy`: `True`
- `search_rollout.extra_params.explore_s1901m_eps`: `0.0`
- `search_rollout.extra_params.explore_f1901m_eps`: `0.0`
- `search_rollout.extra_params.run_do_prob`: `1.0`
- `search_rollout.extra_params.use_ev_targets`: `True`
- `search_rollout.extra_params.use_trained_value`: `True`
- `search_rollout.extra_params.sample_game_json_phases`: `False`
- `search_rollout.extra_params.min_max_episode_movement_phases`: `None`
- `search_rollout.extra_params.max_max_episode_movement_phases`: `None`
- `search_rollout.extra_params.max_training_episode_length`: `None`
- `search_rollout.buffer.capacity`: `10000`
- `search_rollout.enforce_train_gen_ratio`: `9.0`
- `search_rollout.draw_on_stalemate_years`: `3`
- `search_policy_weight`: `0.10000000149011612`
- `bootstrap_offline_targets`: `True`
- `num_train_gpus`: `4`
- `bootstrap_offline_targets`: `True`
- `use_distributed_data_parallel`: `True`
- `launcher.slurm.num_gpus`: `248`
</details>

## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
