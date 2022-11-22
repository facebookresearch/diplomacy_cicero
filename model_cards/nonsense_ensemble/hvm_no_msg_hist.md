# Model 5 model card


## Overview

This is a Transformer-based classifier model used to discriminate between human and model-generated text (where the model generated negatives are provided by a denoising model that was trained on Diplomacy messages), used in an ensemble of nonsense classifiers inside Cicero. It did not condition on message history.

This model was developed by Meta AI. Training began on June 28, 2022.


### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).


### Architecture details

This classifier is based on the [BART](https://arxiv.org/abs/1910.13461)-large model architecture. A linear layer with output dimension 2 is added on top of the hidden states resulting from feeding the encoder states and start token to the decoder, yielding a binary classifier. It was initialized with the weights of a BART-based model further trained to predict dialogue messages (see the paper for a description of the dialogue task). The model predicts between two potential classes: REAL, CORRUPTED.


### Example Input and Output

```
[input]: units: AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, F EDI, F LON; FRANCE: A MAR, A PAR, F BRE; GERMANY: A BER, A MUN, F KIE; ITALY: A ROM, A VEN, F NAP; RUSSIA: A MOS, A WAR, F SEV, F STP/SC; TURKEY: A CON, A SMY, F ANK
centers: AUSTRIA: BUD, TRI, VIE; ENGLAND: EDI, LON, LVP; FRANCE: BRE, MAR, PAR; GERMANY: BER, KIE, MUN; ITALY: NAP, ROM, VEN; RUSSIA: MOS, SEV, STP, WAR; TURKEY: ANK, CON, SMY
GERMANY: A BER KIE; A MUN RUH; F KIE DEN
ENGLAND: A LVP YOR; F EDI NTH; F LON ENG
S1901M ENGLAND ANON 5min WTA PUBLIC NODRAWS:
ENGLAND -> GERMANY: Would you consider a 3 way alliance w Austria if we want to attack France?

[output]: CORRUPTED
```

## Intended Use

This model is for research purposes only. It was used inside the Cicero Diplomacy-playing agent in an ensemble with other nonsense classifiers to detect dialogue messages containing mistakes.


## Datasets used

This model was trained using the WebDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097). We used a generative dialogue model trained for Diplomacy to provide compelling model-generated counterfactuals, like the one seen above ("Would you consider a 3 way alliance w Austria if we want to attack France?").


## Privacy

**Training data:** In order to preserve user privacy, de-identification of user data and automated redaction of personally identifiable information was performed by webDiplomacy prior to being released to the authors of this paper. This automated redaction was verified using a set of 100 games that were hand-redacted by humans, ensuring that the automated scheme achieved 100% recall on these games.

**Deployment:** Furthermore, in live games, the agent accessed webDiplomacy.net through an API that redacted PII on-the-fly following the same protocol.


## Evaluation results

This model filtered 9.47% of messages in live games.

## Related Paper(s)

- BART: https://arxiv.org/abs/1910.13461

## Hyperparameters

<details>
<summary> Hyperparameters </summary>

 - `task`: `orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk`
 - `datatype`: `train`
 - `hide_labels`: `False`
 - `multitask_weights`: `[1]`
 - `batchsize`: `2`
 - `dynamic_batching`: `None`
 - `model`: `bart_classifier`
 - `dict_class`: `parlai.core.dict:DictionaryAgent`
 - `evaltask`: `orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk`
 - `final_extra_opt`: ``
 - `eval_batchsize`: `None`
 - `eval_dynamic_batching`: `None`
 - `num_workers`: `8`
 - `display_examples`: `False`
 - `num_epochs`: `10.0`
 - `max_train_time`: `-1`
 - `max_train_steps`: `75000`
 - `early_stop_at_n_steps`: `-1`
 - `log_every_n_steps`: `100`
 - `validation_every_n_secs`: `-1`
 - `validation_every_n_steps`: `2000`
 - `save_every_n_secs`: `-1`
 - `save_after_valid`: `True`
 - `validation_every_n_epochs`: `-1`
 - `validation_max_exs`: `-1`
 - `short_final_eval`: `False`
 - `validation_patience`: `10`
 - `validation_metric`: `loss`
 - `validation_metric_mode`: `min`
 - `validation_cutoff`: `0.0`
 - `validation_share_agent`: `False`
 - `metrics`: `default`
 - `aggregate_micro`: `False`
 - `dict_maxexs`: `-1`
 - `dict_include_valid`: `False`
 - `dict_include_test`: `False`
 - `log_every_n_secs`: `-1`
 - `distributed_world_size`: `64`
 - `ddp_backend`: `ddp`
 - `image_size`: `256`
 - `image_cropsize`: `224`
 - `model_generated_messages`: `denoising_singleseed`
 - `dialogue_single_turn`: `True`
 - `include_silence_messages`: `False`
 - `calculate_year_metrics`: `False`
 - `calculate_ppl_by_rating_metrics`: `False`
 - `include_sleep_messages`: `False`
 - `output_draw_messages`: `False`
 - `add_sleep_times`: `False`
 - `add_recipient_to_prompt`: `False`
 - `include_style`: `False`
 - `mark_bad_messages`: `None`
 - `filter_bad_messages`: `None`
 - `edit_bad_messages`: `None`
 - `filter_bad_messages_about_draws`: `False`
 - `min_speaker_rating`: `None`
 - `max_game_redacted_words_percent`: `None`
 - `response_view_dialogue_model`: `False`
 - `extend_order_history_since_last_n_movement_phase`: `1`
 - `pseudo_order_generation`: `False`
 - `pseudo_order_generation_future_message`: `True`
 - `pseudo_order_generation_injected_sentence`: `None`
 - `pseudo_order_generation_inject_all`: `True`
 - `pseudo_order_generation_partner_view`: `False`
 - `pseudo_order_generation_current_phase_prefix`: `False`
 - `2person_dialogue`: `False`
 - `all_power_pseudo_orders`: `True`
 - `single_view_pseudo_orders`: `True`
 - `rollout_pseudo_orders`: `True`
 - `rollout_except_movement`: `True`
 - `rollout_phasemajor`: `False`
 - `rollout_actual_orders`: `False`
 - `n_chunks`: `-1`
 - `counting_examples`: `False`
 - `include_task_token`: `False`
 - `message_history_truncation`: `2048`
 - `task_version`: `3`
 - `include_game_info`: `True`
 - `include_player_ratings`: `False`
 - `include_draw_info`: `True`
 - `include_draw_state`: `True`
 - `hide_empty_draw_state`: `True`
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
 - `skip_input_validation`: `False`
 - `input_validation_check_pct`: `0.1`
 - `lie_detector_annotations_dir`: `None`
 - `lie_detector_filter_above_stdev`: `None`
 - `chunk_size`: `80`
 - `beam_size`: `1`
 - `beam_min_length`: `1`
 - `beam_context_block_ngram`: `-1`
 - `beam_block_ngram`: `-1`
 - `beam_block_full_context`: `True`
 - `beam_length_penalty`: `0.65`
 - `skip_generation`: `True`
 - `topp`: `0.9`
 - `beam_delay`: `30`
 - `beam_block_list_filename`: `None`
 - `temperature`: `1.0`
 - `compute_tokenized_bleu`: `False`
 - `candidates`: `inline`
 - `eval_candidates`: `inline`
 - `interactive_candidates`: `fixed`
 - `repeat_blocking_heuristic`: `True`
 - `fixed_candidates_path`: `None`
 - `fixed_candidate_vecs`: `reuse`
 - `encode_candidate_vecs`: `True`
 - `encode_candidate_vecs_batchsize`: `256`
 - `train_predict`: `False`
 - `cap_num_predictions`: `100`
 - `ignore_bad_candidates`: `False`
 - `rank_top_k`: `-1`
 - `inference`: `max`
 - `topk`: `5`
 - `return_cand_scores`: `False`
 - `embedding_size`: `1024`
 - `n_layers`: `2`
 - `ffn_size`: `4096`
 - `dropout`: `0.1`
 - `attention_dropout`: `0.0`
 - `relu_dropout`: `0.0`
 - `n_heads`: `16`
 - `learn_positional_embeddings`: `True`
 - `embeddings_scale`: `False`
 - `n_positions`: `2048`
 - `n_segments`: `0`
 - `variant`: `bart`
 - `activation`: `gelu`
 - `output_scaling`: `1.0`
 - `n_encoder_layers`: `12`
 - `n_decoder_layers`: `12`
 - `model_parallel`: `False`
 - `checkpoint_activations`: `False`
 - `use_memories`: `False`
 - `wrap_memory_encoder`: `False`
 - `memory_attention`: `sqrt`
 - `normalize_sent_emb`: `False`
 - `share_encoders`: `True`
 - `share_word_embeddings`: `True`
 - `learn_embeddings`: `True`
 - `reduction_type`: `first`
 - `embedding_type`: `random`
 - `embedding_projection`: `random`
 - `fp16`: `True`
 - `fp16_impl`: `mem_efficient`
 - `force_fp16_tokens`: `True`
 - `optimizer`: `mem_eff_adam`
 - `learningrate`: `5e-05`
 - `gradient_clip`: `0.1`
 - `adam_eps`: `1e-08`
 - `adafactor_eps`: `[1e-30, 0.001]`
 - `momentum`: `0`
 - `nesterov`: `True`
 - `nus`: `[0.7]`
 - `betas`: `[0.9, 0.999]`
 - `weight_decay`: `None`
 - `rank_candidates`: `False`
 - `truncate`: `1024`
 - `text_truncate`: `2048`
 - `label_truncate`: `10`
 - `history_reversed`: `False`
 - `history_size`: `-1`
 - `person_tokens`: `False`
 - `split_lines`: `False`
 - `use_reply`: `none`
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
 - `classes`: `['REAL', 'CORRUPTED']`
 - `class_weights`: `None`
 - `ref_class`: `None`
 - `threshold`: `0.5`
 - `print_scores`: `False`
 - `data_parallel`: `False`
 - `classes_from_file`: `None`
 - `ignore_labels`: `None`
 - `update_classifier_head_only`: `False`
 - `load_from_pretrained_ranker`: `False`
 - `freeze_enc_dec_weights`: `False`
 - `starttime`: `Jun28_15-52`
 - `rank`: `0`
</details>



## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).