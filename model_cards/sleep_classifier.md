# sleep_classifier model card

## Overview

The sleep classifier is used by Cicero to determine whether it should send a message to a given recipient, and if so, when to send it. Specifically, it models the lengths of time between messages in the webDiplomacy dataset, rounded to a set of hard-coded lengths of time (a special "INF" class covers the case where no message was sent.

This model was developed by Meta AI. Training began on March 23, 2022.


### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).


### Architecture details

This is a 400M parameter BART-based Transformer encoder-decoder model with an added linear layer producing logits for 19 time classes.

Additional hyperparameter details can be found below.

### Example Input and Output

```
[input]:
S1901M
0 ENGLAND -> AUSTRIA: Hey Austria! Best of luck this game, let me know if there's anything you'd like to discuss!
0 ENGLAND -> RUSSIA: Hi Russia! ~N~ Just wanted to establish comms and see how you wanted to handle Scandinavia?

units: AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, F EDI, F LON; FRANCE: A MAR, A PAR, F BRE; GERMANY: A BER, A MUN, F KIE; ITALY: A ROM, A VEN, F NAP; RUSSIA: A MOS, A WAR, F SEV, F STP/SC; TURKEY: A CON, A SMY, F ANK
centers: AUSTRIA: BUD, TRI, VIE; ENGLAND: EDI, LON, LVP; FRANCE: BRE, MAR, PAR; GERMANY: BER, KIE, MUN; ITALY: NAP, ROM, VEN; RUSSIA: MOS, SEV, STP, WAR; TURKEY: ANK, CON, SMY
0
S1901M ENGLAND 5 ANON 5min SOS PUBLIC HASDRAWS for AUSTRIA:

[output]: probabilities for 18 time classes and one "inf" class
```

## Intended Use

This model is for research purposes only. It was used inside the Cicero Diplomacy-playing agent for choosing message recipients and scheduling messages.

## Limitations

The webDiplomacy dataset does not contain timestamps for phase transitions, and so the exact "ground truth" scheduling is not known for many messages in the dataset.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097).

## Privacy

**Training data:** In order to preserve user privacy, de-identification of user data and automated redaction of personally identifiable information was performed by webDiplomacy prior to being released to the authors of this paper. This automated redaction was verified using a set of 100 games that were hand-redacted by humans, ensuring that the automated scheme achieved 100% recall on these games.

**Deployment:** Furthermore, in live games, the agent accessed webDiplomacy.net through an API that redacted PII on-the-fly following the same protocol.


## Related Paper(s)

BART: https://arxiv.org/abs/1910.13461

## Hyperparameters

<details>
<summary> Hyperparameters </summary>

 - `task`: `message_history_orderhistorysincelastmovementphase_shortstate_sleepsix_chunk`
 - `datatype`: `train`
 - `hide_labels`: `False`
 - `multitask_weights`: `[1]`
 - `batchsize`: `2`
 - `dynamic_batching`: `None`
 - `model`: `bart_marginal_likelihood`
 - `dict_class`: `parlai.core.dict:DictionaryAgent`
 - `evaltask`: `None`
 - `final_extra_opt`: ``
 - `eval_batchsize`: `None`
 - `eval_dynamic_batching`: `None`
 - `num_workers`: `0`
 - `display_examples`: `False`
 - `num_epochs`: `10.0`
 - `max_train_time`: `-1`
 - `max_train_steps`: `10000`
 - `log_every_n_steps`: `100`
 - `validation_every_n_secs`: `-1`
 - `validation_every_n_steps`: `500`
 - `save_every_n_secs`: `-1`
 - `save_after_valid`: `True`
 - `validation_every_n_epochs`: `-1`
 - `validation_max_exs`: `-1`
 - `short_final_eval`: `False`
 - `validation_patience`: `50`
 - `validation_metric`: `loss`
 - `validation_metric_mode`: `min`
 - `validation_cutoff`: `0.0`
 - `validation_share_agent`: `False`
 - `metrics`: `default`
 - `aggregate_micro`: `False`
 - `dict_maxexs`: `-1`
 - `dict_include_valid`: `False`
 - `dict_include_test`: `False`
 - `log_every_n_secs`: `300.0`
 - `distributed_world_size`: `128`
 - `ddp_backend`: `ddp`
 - `image_size`: `256`
 - `image_cropsize`: `224`
 - `n_chunks`: `-1`
 - `counting_examples`: `False`
 - `include_task_token`: `False`
 - `message_history_truncation`: `2048`
 - `task_version`: `3`
 - `include_game_info`: `True`
 - `include_player_ratings`: `True`
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
 - `input_validation_check_pct`: `0.0`
 - `lie_detector_annotations_dir`: `None`
 - `lie_detector_filter_above_stdev`: `None`
 - `chunk_size`: `40`
 - `beam_size`: `1`
 - `beam_min_length`: `1`
 - `beam_context_block_ngram`: `-1`
 - `beam_block_ngram`: `-1`
 - `beam_block_full_context`: `True`
 - `beam_length_penalty`: `0.65`
 - `skip_generation`: `False`
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
 - `attention_dropout`: `0.1`
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
 - `special_tok_lst`: `None`
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
 - `warmup_updates`: `1000`
 - `warmup_rate`: `0.0001`
 - `update_freq`: `32`
 - `classes`: `['pad', '0', '15', '35', '64', '113', '207', '385', '698', '1220', '2060', '3340', '5300', '8390', '13500', '22600', '39400', '86000', 'inf']`
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
 - `include_sleep_messages`: `True`
 - `add_sleep_times`: `True`
 - `starttime`: `Mar23_12-25`
 - `rank`: `0`
</details>


## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
