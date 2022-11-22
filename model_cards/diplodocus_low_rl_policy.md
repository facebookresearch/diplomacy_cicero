# diplodocus_low_rl_policy.ckpt model card

## Overview

This model was the RL policy for Diplodocus-low. It uses a transformer encoder on a representation of the gamestate with an LSTM decoder to autogressively decode the action to be predicted.

This model was developed by Meta AI. It was trained in Decenmber 2021.

### Paper link

Find more details in our paper [here](https://arxiv.org/abs/2210.05492).

### Architecture details

This model has roughly 8 million trainable parameters. See Appendix F [here](https://arxiv.org/abs/2210.05492) for details on the architecture and input and output encoding.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It also decodes only one player's action at a time, predicting its probability independently from that of any other player.

## Datasets used

This model was trained via RL with regularization towards other models trained on the webDiplomacy dataset, described in the [paper](https://arxiv.org/abs/2210.05492).

## Privacy

This model does not observe any personally identifiable information.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `lstm_dropout`: `0.0`
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
 - `has_policy`: `True`
 - `has_value`: `False`
 - `use_year`: `True`
 - `critic_weight`: `1.0`
- `optimizer.grad_clip`: `0.5`
- `optimizer.warmup_epochs`: `100`
- `optimizer.adam.lr`: `0.0001`
- `discounting`: `1.0`
- `model_path`: `models/nopress_human_imitation_for_rl_1.ckpt`
- `value_model_path`: `models/nopress_human_imitation_for_rl_1.ckpt`
- `search_rollout.agent.searchbot.n_rollouts`: `256`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.use_final_iter`: `False`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.plausible_orders_cfg.n_plausible_orders`: `50`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.plausible_orders_cfg.max_actions_units_ratio`: `6.0`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.plausible_orders_cfg.req_size`: `250`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.rollouts_cfg.year_spring_prob_of_ending`: `1901,0;1909,0.2;1914,0.4`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.qre`: `{'eta': 10.0, 'target_pi': 'BLUEPRINT'}`
- `search_rollout.agent.bqre1p.base_searchbot_cfg.half_precision`: `True`
- `search_rollout.agent.bqre1p.num_player_types`: `2`
- `search_rollout.agent.bqre1p.agent_type`: `1`
- `search_rollout.agent.bqre1p.player_types`: `{'log_uniform': {'min_lambda': 0.0001, 'max_lambda': 0.1}}`
- `search_rollout.agent.bqre1p.player_types.policies.model_path`: `models/nopress_human_imitation_for_rl_2.ckpt`
- `search_rollout.num_workers_per_gpu`: `8`
- `search_rollout.chunk_length`: `128`
- `search_rollout.batch_size`: `8`
- `search_rollout.extra_params.explore_eps`: `0.10000000149011612`
- `search_rollout.extra_params.independent_explore`: `True`
- `search_rollout.extra_params.use_trained_policy`: `True`
- `search_rollout.extra_params.do`: `{'max_iters': 14, 'min_diff': 0.03999999910593033, 'min_diff_percentage': 10.0, 'max_op_actions': 10, 'use_exact_op_policy': False, 'shuffle_powers': True, 'generation': {'max_actions': 1000, 'base_strategy_model': {}, 'local_uniform': {'num_base_actions': 5, 'use_search_policy': True, 'fix_uncoordinated_base': True, 'with_holes': True}}}`
- `search_rollout.extra_params.explore_s1901m_eps`: `0.0`
- `search_rollout.extra_params.explore_f1901m_eps`: `0.0`
- `search_rollout.extra_params.run_do_prob`: `0.10000000149011612`
- `search_rollout.extra_params.use_ev_targets`: `True`
- `search_rollout.extra_params.use_trained_value`: `True`
- `search_rollout.extra_params.sample_game_json_phases`: `False`
- `search_rollout.buffer.capacity`: `10000`
- `search_rollout.enforce_train_gen_ratio`: `6.0`
- `search_rollout.draw_on_stalemate_years`: `3`
- `search_policy_weight`: `0.10000000149011612`
- `bootstrap_offline_targets`: `True`
- `num_train_gpus`: `4`
- `use_distributed_data_parallel`: `True`
- `launcher.slurm.num_gpus`: `512`
</details>




## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
