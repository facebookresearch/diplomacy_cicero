# cicero_imitation_bilateral_orders_prefix model card

## Overview

This model is used to predict the orders that a power B will play in the current phase, given the game state and either (a) a "prefix" of the dialogue from the perspective of power A (i.e. all the dialogue up to some point in time in the phase), or (b) a prefix of only the dialogue between A and B (in which case the prompt includes the phrase "two powers"). In Cicero, it's used to sample or likelihood-reweight a truncated distribution of order strings to produce a 'blueprint policy' that regularizes the search procedure. In retreat and build phases it predicts orders through the next movement phase.

The model uses a standard [pre-trained BART model](https://arxiv.org/pdf/1910.13461.pdf). The model was first fine-tuned on the Diplomacy dialogue dataset, and then further fine-tuned on the order prediction task.

This model was developed by Meta AI. It was trained in March 2022.

### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).

### Architecture details

This is a standard [pre-trained BART model](https://arxiv.org/pdf/1910.13461.pdf).

### Example Input and Output
```
[input]: S1901M
0 FRANCE -> ITALY: DMZ PIE?
...
32 ITALY -> FRANCE: Sounds good.
DRAWS:
units: AUSTRIA: A BUD, A VIE, F TRI; ENGLAND: A LVP, F EDI, F LON; FRANCE: A MAR, A PAR, F BRE; GERMANY: A BER, A MUN, F KIE; ITALY: A ROM, A VEN, F NAP; RUSSIA: A MOS, A WAR, F SEV, F STP/SC; TURKEY: A CON, A SMY, F ANK
centers: AUSTRIA: BUD, TRI, VIE; ENGLAND: EDI, LON, LVP; FRANCE: BRE, MAR, PAR; GERMANY: BER, KIE, MUN; ITALY: NAP, ROM, VEN; RUSSIA: MOS, SEV, STP, WAR; TURKEY: ANK, CON, SMY
GERMANY: A BER KIE; A MUN RUH; F KIE DEN
F1901M
A KIE HOL; A RUH BEL; F DEN SWE
FRANCE: A MAR S A PAR BUR; A PAR BUR; F BRE MAO
F1901M
A BUR BEL; A MAR SPA; F MAO POR
8
S1901M FRANCE 5 ANON 5min WTA two powers for ITALY:
[output]: A MAR S A PAR BUR; A PAR BUR; F BRE MAO
```

## Intended Use

This model is for research purposes only.

## Limitations

This model only predicts orders for a single power independently (although conditioned on another power's dialogue). The model can only predict the orders for the current phase.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097).

## Privacy

This model was trained on anonymized Diplomacy dialogue, but only outputs order strings.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `task`: `message_history_orderhistorysincelastmovementphase_shortstate_allorderindependentrollout_chunk`
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
 - `max_train_steps`: `250000`
 - `log_every_n_steps`: `100`
 - `validation_every_n_secs`: `-1`
 - `validation_every_n_steps`: `1000`
 - `save_every_n_secs`: `3600.0`
 - `save_after_valid`: `True`
 - `validation_every_n_epochs`: `-1`
 - `validation_max_exs`: `-1`
 - `short_final_eval`: `False`
 - `validation_patience`: `30`
 - `validation_metric`: `ppl`
 - `validation_metric_mode`: `min`
 - `validation_cutoff`: `1.0`
 - `validation_share_agent`: `False`
 - `metrics`: `default`
 - `aggregate_micro`: `False`
 - `dict_maxexs`: `-1`
 - `dict_include_valid`: `False`
 - `dict_include_test`: `False`
 - `log_every_n_secs`: `120.0`
 - `distributed_world_size`: `128`
 - `ddp_backend`: `ddp`
 - `image_size`: `256`
 - `image_cropsize`: `224`
 - `allorders_mark_all_holds`: `True`
 - `filter_all_holds`: `True`
 - `n_chunks`: `-1`
 - `counting_examples`: `False`
 - `include_task_token`: `False`
 - `message_history_truncation`: `2048`
 - `task_version`: `3`
 - `include_game_info`: `True`
 - `include_player_ratings`: `True`
 - `include_draw_info`: `False`
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
 - `share_word_embeddings`: `True`
 - `n_encoder_layers`: `12`
 - `n_decoder_layers`: `12`
 - `model_parallel`: `False`
 - `checkpoint_activations`: `False`
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
 - `fp16_impl`: `mem_efficient`
 - `force_fp16_tokens`: `True`
 - `optimizer`: `mem_eff_adam`
 - `learningrate`: `8e-05`
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
 - `special_tok_lst`: `NON_SILENCE,[EO_STATE],[REDACTED],Austria,England,Germany,AUSTRIA,ENGLAND,GERMANY,SILENCE,France,Russia,Turkey,FRANCE,RUSSIA,TURKEY,SPA/NC,STP/SC,BUL/SC,STP/NC,BUL/EC,SPA/SC,[EO_O],[EO_M],Italy,ITALY,VEN,ALB,KIE,BAR,NWG,TUS,EDI,GRE,PRU,BUD,HEL,IRI,SKA,GAL,TYS,RUM,NAP,SMY,LON,ADR,BOH,EAS,BEL,ANK,MAR,APU,TUN,PIE,SPA,HOL,SIL,MUN,YOR,LYO,ION,TYR,CON,WES,ENG,NAF,UKR,AEG,SER,ROM,WAR,BUR,VIA,VIE,LVP,GAS,BAL,BUL,BLA,TRI,ARM,SWE,RUH,NTH,NWY,BOT,DEN,NAO,WAL,BER,PIC,MOS,STP,BRE,PAR,SEV,MAO,SYR,FIN,LVN,CLY,POR`
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
 - `starttime`: `Mar05_16-31`
 - `rank`: `0`
</details>


## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
