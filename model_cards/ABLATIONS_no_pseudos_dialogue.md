# Dialogue (ABLATIONS -- no intents) model card


## Overview

This is a generative Transformer model used to generate dialogue message candidates, used for ablations in our paper. In particular, it is not grounded in intents.

This model was developed by Meta AI. Training began on August 29, 2022.


### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).


### Architecture details

This is a 2.7B parameter R2C2-based Transformer encoder-decoder model.


### Example Input and Output

```
[input]: S1901M
0 ITALY -> ALL: Hi everyone GL!
...
32 ITALY -> FRANCE: What do you want to do this game?
units: AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, F EDI, F LON; FRANCE: A MAR, A PAR, F BRE; GERMANY: A BER, A MUN, F KIE; ITALY: A ROM, A VEN, F NAP; RUSSIA: A MOS, A WAR, F SEV, F STP/SC; TURKEY: A CON, A SMY, F ANK
centers: AUSTRIA: BUD, TRI, VIE; ENGLAND: EDI, LON, LVP; FRANCE: BRE, MAR, PAR; GERMANY: BER, KIE, MUN; ITALY: NAP, ROM, VEN; RUSSIA: MOS, SEV, STP, WAR; TURKEY: ANK, CON, SMY
18
S1901M FRANCE -> GERMANY 5 ANON 5min WTA PUBLIC NODRAWS:

[output]: GERMANY: I'm not sure yet, what do you have in mind?
```

## Intended Use

This model is for research purposes only. It was used for generating dialogue to compare to the Cicero dialogue model.

## Limitations

This model is limited in a number of ways. It occasionally generates nonsense or hallucinates impossible moves. It also occasionally generates toxic content. It should not be used outside of the Cicero Diplomacy agent. See paper for more details.


## Datasets used

This model was trained using the WebDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097).


## Privacy

**Training data:** In order to preserve user privacy, de-identification of user data and automated redaction of personally identifiable information was performed by webDiplomacy prior to being released to the authors of this paper. This automated redaction was verified using a set of 100 games that were hand-redacted by humans, ensuring that the automated scheme achieved 100% recall on these games.

**Deployment:** Furthermore, in live games, the agent accessed webDiplomacy.net through an API that redacted PII on-the-fly following the same protocol.


## Evaluation results

This model achieves a validation PPL of 8.042 on the task `message_history_orderhistorysincelastmovementphase_shortstate_dialogue_chunk` (with other task parameters defined below)

## Related Paper(s)

- BART: https://arxiv.org/abs/1910.13461
- R2C2: https://arxiv.org/pdf/2203.13224.pdf

## Hyperparameters


<details>
<summary> Hyperparameters </summary>

 - `task`: `message_history_orderhistorysincelastmovementphase_shortstate_dialogue_chunk`
 - `datatype`: `train`
 - `hide_labels`: `False`
 - `multitask_weights`: `[1]`
 - `batchsize`: `2`
 - `dynamic_batching`: `None`
 - `model`: `bart`
 - `dict_class`: `parlai.core.dict:DictionaryAgent`
 - `evaltask`: `None`
 - `final_extra_opt`: ``
 - `eval_batchsize`: `None`
 - `eval_dynamic_batching`: `None`
 - `num_workers`: `8`
 - `display_examples`: `False`
 - `num_epochs`: `-1`
 - `max_train_time`: `-1`
 - `max_train_steps`: `100000`
 - `log_every_n_steps`: `100`
 - `validation_every_n_secs`: `-1`
 - `validation_every_n_steps`: `1000`
 - `save_every_n_secs`: `-1`
 - `save_after_valid`: `True`
 - `validation_every_n_epochs`: `-1`
 - `validation_max_exs`: `-1`
 - `short_final_eval`: `False`
 - `validation_patience`: `100`
 - `validation_metric`: `ppl`
 - `validation_metric_mode`: `min`
 - `validation_cutoff`: `1.0`
 - `validation_share_agent`: `False`
 - `metrics`: `default`
 - `aggregate_micro`: `False`
 - `dict_maxexs`: `-1`
 - `dict_include_valid`: `False`
 - `dict_include_test`: `False`
 - `log_every_n_secs`: `-1`
 - `distributed_world_size`: `256`
 - `ddp_backend`: `zero2`
 - `image_size`: `256`
 - `image_cropsize`: `224`
 - `dialogue_single_turn`: `True`
 - `include_silence_messages`: `False`
 - `calculate_year_metrics`: `False`
 - `calculate_ppl_by_rating_metrics`: `False`
 - `include_sleep_messages`: `False`
 - `output_draw_messages`: `False`
 - `add_sleep_times`: `True`
 - `add_recipient_to_prompt`: `True`
 - `include_style`: `False`
 - `mark_bad_messages`: `phase_repeats,offensive_language,redacted`
 - `filter_bad_messages`: `draws`
 - `edit_bad_messages`: `None`
 - `filter_bad_messages_about_draws`: `False`
 - `min_speaker_rating`: `None`
 - `max_game_redacted_words_percent`: `None`
 - `response_view_dialogue_model`: `False`
 - `extend_order_history_since_last_n_movement_phase`: `1`
 - `extend_state_history_since_last_n_movement_phase`: `0`
 - `pseudo_order_generation`: `False`
 - `pseudo_order_generation_future_message`: `True`
 - `pseudo_order_generation_injected_sentence`: `None`
 - `pseudo_order_generation_inject_all`: `True`
 - `pseudo_order_generation_partner_view`: `False`
 - `pseudo_order_generation_current_phase_prefix`: `False`
 - `two_party_dialogue`: `False`
 - `no_speaker_dialogue_history`: `False`
 - `remove_n_latest_messages_from_dialogue_history`: `None`
 - `all_power_pseudo_orders`: `True`
 - `single_view_pseudo_orders`: `False`
 - `rollout_pseudo_orders`: `False`
 - `rollout_except_movement`: `True`
 - `rollout_phasemajor`: `False`
 - `rollout_actual_orders`: `False`
 - `n_chunks`: `-1`
 - `counting_examples`: `False`
 - `include_task_token`: `False`
 - `message_history_truncation`: `2048`
 - `task_version`: `3`
 - `include_game_info`: `True`
 - `include_player_ratings`: `True`
 - `include_draw_info`: `True`
 - `include_draw_state`: `True`
 - `hide_empty_draw_state`: `False`
 - `include_centers_state`: `True`
 - `include_builds_state`: `False`
 - `player_rating_max`: `5`
 - `player_rating_percentiles`: `games_played`
 - `set_player_rating`: `-1`
 - `include_player_chattiness`: `False`
 - `set_player_chattiness`: `-1`
 - `only_phase`: `None`
 - `only_game_id`: `None`
 - `only_chunk`: `-1`
 - `skip_input_validation`: `True`
 - `input_validation_check_pct`: `0.1`
 - `lie_detector_annotations_dir`: `None`
 - `lie_detector_filter_above_stdev`: `None`
 - `counterfactual_game_cache`: `0`
 - `chunk_size`: `80`
 - `embedding_size`: `2048`
 - `n_layers`: `22`
 - `ffn_size`: `8192`
 - `dropout`: `0.1`
 - `attention_dropout`: `0.0`
 - `relu_dropout`: `0.0`
 - `n_heads`: `32`
 - `learn_positional_embeddings`: `True`
 - `embeddings_scale`: `True`
 - `n_positions`: `2048`
 - `n_segments`: `0`
 - `variant`: `prelayernorm`
 - `activation`: `gelu`
 - `output_scaling`: `1.0`
 - `share_word_embeddings`: `True`
 - `n_encoder_layers`: `22`
 - `n_decoder_layers`: `22`
 - `model_parallel`: `False`
 - `checkpoint_activations`: `True`
 - `beam_size`: `1`
 - `beam_min_length`: `1`
 - `beam_context_block_ngram`: `-1`
 - `beam_block_ngram`: `-1`
 - `beam_block_full_context`: `True`
 - `beam_length_penalty`: `0.65`
 - `skip_generation`: `True`
 - `inference`: `greedy`
 - `topk`: `10`
 - `topp`: `0.9`
 - `beam_delay`: `30`
 - `beam_block_list_filename`: `None`
 - `temperature`: `1.0`
 - `compute_tokenized_bleu`: `False`
 - `embedding_type`: `random`
 - `embedding_projection`: `random`
 - `fp16`: `True`
 - `fp16_impl`: `safe`
 - `force_fp16_tokens`: `True`
 - `optimizer`: `adam`
 - `learningrate`: `0.0001`
 - `gradient_clip`: `0.1`
 - `adam_eps`: `1e-08`
 - `adafactor_eps`: `[1e-30, 0.001]`
 - `momentum`: `0`
 - `nesterov`: `True`
 - `nus`: `[0.7]`
 - `betas`: `[0.9, 0.999]`
 - `weight_decay`: `None`
 - `rank_candidates`: `False`
 - `truncate`: `-1`
 - `text_truncate`: `2048`
 - `label_truncate`: `512`
 - `history_reversed`: `False`
 - `history_size`: `-1`
 - `person_tokens`: `False`
 - `split_lines`: `False`
 - `use_reply`: `label`
 - `add_p1_after_newln`: `False`
 - `history_add_global_end_token`: `None`
 - `special_tok_lst`: `[REDACTED],NON-ANON,HASDRAWS,Austria,England,Germany,AUSTRIA,ENGLAND,GERMANY,ALL-UNK,PRIVATE,NODRAWS,France,Russia,Turkey,FRANCE,RUSSIA,TURKEY,SPA/NC,STP/SC,BUL/SC,STP/NC,BUL/EC,SPA/SC,PUBLIC,Italy,ITALY,ANON,PPSC,VEN,ALB,KIE,BAR,NWG,TUS,EDI,GRE,PRU,BUD,HEL,IRI,SKA,GAL,TYS,RUM,NAP,SMY,LON,ADR,BOH,EAS,BEL,ANK,MAR,APU,TUN,PIE,SPA,HOL,SIL,MUN,YOR,LYO,ION,TYR,CON,WES,ENG,NAF,UKR,AEG,SER,ROM,WAR,BUR,VIA,VIE,LVP,GAS,BAL,BUL,BLA,TRI,ARM,SWE,RUH,NTH,NWY,BOT,DEN,NAO,WAL,BER,PIC,MOS,STP,BRE,PAR,SEV,MAO,SYR,FIN,LVN,CLY,POR,BAD,SOS,WTA,->`
 - `gpu`: `0`
 - `no_cuda`: `False`
 - `dict_initpath`: `None`
 - `dict_language`: `english`
 - `dict_max_ngram_size`: `-1`
 - `dict_minfreq`: `0`
 - `dict_maxtokens`: `-1`
 - `dict_nulltoken`: `__null__`
 - `dict_starttoken`: `__start__`
 - `dict_endtoken`: `__end__`
 - `dict_unktoken`: `__unk__`
 - `dict_tokenizer`: `gpt2`
 - `dict_lower`: `False`
 - `bpe_debug`: `False`
 - `dict_textfields`: `text,labels`
 - `bpe_vocab`: `None`
 - `bpe_merge`: `None`
 - `bpe_add_prefix_space`: `None`
 - `bpe_dropout`: `None`
 - `lr_scheduler`: `linear`
 - `lr_scheduler_patience`: `3`
 - `lr_scheduler_decay`: `0.5`
 - `invsqrt_lr_decay_gamma`: `-1`
 - `warmup_updates`: `8000`
 - `warmup_rate`: `0.0001`
 - `update_freq`: `1`
 - `starttime`: `Aug29_15-23`
 - `rank`: `0`
</details>


## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
