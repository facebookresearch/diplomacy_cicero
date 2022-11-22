# Model 11 model card


## Overview

This is a Transformer-based classifier model used to discriminate between human and model-generated text, used in an ensemble of nonsense classifiers inside Cicero. The model generated negatives are provided by a "weak" generative dialogue model fine-tuned for Diplomacy that transforms examples with words suggesting the presence of a justification (e.g. "because") by replacing the justification (e.g text after "because") with a model-generated continuation.

This model was developed by Meta AI. Training began on 2022-07-29.


### Paper link

Find more details in our paper [here](https://www.science.org/doi/10.1126/science.ade9097).


### Architecture details

This classifier is based on the [BART](https://arxiv.org/abs/1910.13461)-large model architecture. A linear layer with output dimension 2 is added on top of the hidden states resulting from feeding the encoder states and start token to the decoder, yielding a binary classifier. It was initialized with the weights of a BART-based model further trained to predict dialogue messages (see the paper for a description of the dialogue task). The model predicts between two potential classes: REAL, CORRUPTED.


### Example Input and Output

_[input]_
F1902M
[...]
S1903M
[...]
F1903M
[...]
S1904M
[...]
RUSSIA -> GERMANY: Work with me?
S1903M
ENGLAND: A BRE S A BUR PAR; A STP FIN; F ENG NTH; F MAO S F WES SPA; F NTH NWG
FRANCE: A BEL RUH; A MAR PIE; F LYO SPA/SC; F POR S F LYO SPA/SC
ITALY: A PIE TUS; A ROM APU; F WES SPA/SC
GERMANY: A BOH MUN; A BUR PAR; A MUN BUR; A RUH BEL; F DEN KIE
AUSTRIA: A SER GRE; A TRI S A VEN; A VEN S F ADR APU; A VIE H; F ADR APU; F AEG ION
TURKEY: A CON H; F ANK H; F BLA H
RUSSIA: A BUL H; A MOS STP; A RUM S A BUL; A SEV H; A WAR LVN; F NWY S A MOS STP; F SWE S F NWY
F1903M
ENGLAND: A BRE GAS; A FIN NWY; F MAO POR; F NTH S F NWY; F NWG BAR
FRANCE: A BEL HOL; A PIE TYR; F LYO SPA/SC; F POR S F LYO SPA/SC
ITALY: A ROM H; A TUS PIE; F WES SPA/SC
GERMANY: A BUR S A RUH BEL; A MUN RUH; A PAR H; A RUH BEL; F KIE HOL
AUSTRIA: A GRE SER; A TRI S A VEN; A VEN H; A VIE H; F APU NAP; F ION S F APU NAP
TURKEY: A CON ANK; F ANK BLA; F BLA CON
RUSSIA: A BUL H; A LVN H; A RUM S A BUL; A SEV H; A STP S F NWY; F NWY H; F SWE S F NWY
F1903R
FRANCE: A BEL R PIC
W1903A
ENGLAND: A FIN D
FRANCE: A TYR D
ITALY: A PIE D
GERMANY: A MUN B; F BER B
AUSTRIA: A BUD B
RUSSIA: A MOS B
F1903M
units: AUSTRIA: A GRE, A TRI, A VEN, A VIE, F APU, F ION; ENGLAND: A BRE, A FIN, F MAO, F NTH, F NWG; FRANCE: A BEL, A PIE, F LYO, F POR; GERMANY: A BUR, A MUN, A PAR, A RUH, F KIE; ITALY: A ROM, A TUS, F WES; RUSSIA: A BUL, A LVN, A RUM, A SEV, A STP, F NWY, F SWE; TURKEY: A CON, F ANK, F BLA
centers: AUSTRIA: BUD, GRE, SER, TRI, VEN, VIE; ENGLAND: BRE, EDI, LON, LVP, STP; FRANCE: BEL, MAR, PAR, POR, SPA; GERMANY: BER, DEN, HOL, KIE, MUN; ITALY: NAP, ROM, TUN; RUSSIA: BUL, MOS, NWY, RUM, SEV, SWE, WAR; TURKEY: ANK, CON, SMY
F1903R
units: AUSTRIA: A SER, A TRI, A VEN, A VIE, F ION, F NAP; ENGLAND: A FIN, A GAS, F BAR, F MAO, F NTH; FRANCE: *A BEL, A TYR, F LYO, F POR; GERMANY: A BEL, A BUR, A PAR, A RUH, F KIE; ITALY: A PIE, A ROM, F WES; RUSSIA: A BUL, A LVN, A RUM, A SEV, A STP, F NWY, F SWE; TURKEY: A ANK, F BLA, F CON
retreats: FRANCE: A BEL - PIC
centers: AUSTRIA: BUD, GRE, SER, TRI, VEN, VIE; ENGLAND: BRE, EDI, LON, LVP, STP; FRANCE: BEL, MAR, PAR, POR, SPA; GERMANY: BER, DEN, HOL, KIE, MUN; ITALY: NAP, ROM, TUN; RUSSIA: BUL, MOS, NWY, RUM, SEV, SWE, WAR; TURKEY: ANK, CON, SMY
W1903A
units: AUSTRIA: A SER, A TRI, A VEN, A VIE, F ION, F NAP; ENGLAND: A FIN, A GAS, F BAR, F MAO, F NTH; FRANCE: A PIC, A TYR, F LYO, F POR; GERMANY: A BEL, A BUR, A PAR, A RUH, F KIE; ITALY: A PIE, A ROM, F WES; RUSSIA: A BUL, A LVN, A RUM, A SEV, A STP, F NWY, F SWE; TURKEY: A ANK, F BLA, F CON
centers: AUSTRIA: BUD, GRE, NAP, SER, TRI, VEN, VIE; ENGLAND: BRE, EDI, LON, LVP; FRANCE: MAR, POR, SPA; GERMANY: BEL, BER, DEN, HOL, KIE, MUN, PAR; ITALY: ROM, TUN; RUSSIA: BUL, MOS, NWY, RUM, SEV, STP, SWE, WAR; TURKEY: ANK, CON, SMY
S1904M
units: AUSTRIA: A BUD, A SER, A TRI, A VEN, A VIE, F ION, F NAP; ENGLAND: A GAS, F BAR, F MAO, F NTH; FRANCE: A PIC, F LYO, F POR; GERMANY: A BEL, A BUR, A MUN, A PAR, A RUH, F BER, F KIE; ITALY: A ROM, F WES; RUSSIA: A BUL, A LVN, A MOS, A RUM, A SEV, A STP, F NWY, F SWE; TURKEY: A ANK, F BLA, F CON
centers: AUSTRIA: BUD, GRE, NAP, SER, TRI, VEN, VIE; ENGLAND: BRE, EDI, LON, LVP; FRANCE: MAR, POR, SPA; GERMANY: BEL, BER, DEN, HOL, KIE, MUN, PAR; ITALY: ROM, TUN; RUSSIA: BUL, MOS, NWY, RUM, SEV, STP, SWE, WAR; TURKEY: ANK, CON, SMY
RUSSIA: A BUL H; A LVN H; A MOS SEV; A RUM S A BUL; A SEV ARM; A STP S F NWY; F NWY H; F SWE S F NWY
GERMANY: A BEL PIC; A BUR MAR; A MUN BUR; A PAR S A BEL PIC; A RUH BEL; F BER KIE; F KIE HOL
S1904M GERMANY NON-ANON 1440min PPSC ALL-UNK PUBLIC NODRAWS:
GERMANY -> RUSSIA: I'd consider it because I think you're smart.

_[output]_
CORRUPTED

## Intended Use

This model is for research purposes only. It was used inside the Cicero Diplomacy-playing agent in an ensemble with other nonsense classifiers to detect dialogue messages containing mistakes.


## Datasets used

This model was trained using the WebDiplomacy dataset, described in the [paper](https://www.science.org/doi/10.1126/science.ade9097). We used a generative dialogue model trained for Diplomacy to provide compelling model-generated counterfactuals, like the one seen above ("Would you consider a 3 way alliance w Austria if we want to attack France?").


## Privacy

**Training data:** In order to preserve user privacy, de-identification of user data and automated redaction of personally identifiable information was performed by webDiplomacy prior to being released to the authors of this paper. This automated redaction was verified using a set of 100 games that were hand-redacted by humans, ensuring that the automated scheme achieved 100% recall on these games.

**Deployment:** Furthermore, in live games, the agent accessed webDiplomacy.net through an API that redacted PII on-the-fly following the same protocol.


## Evaluation results

This model filtered 18.77% of messages in live games.

## Related Paper(s)

- BART: https://arxiv.org/abs/1910.13461

## Hyperparameters

<details>
<summary> Hyperparameters </summary>

 - `task`: `message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk`
 - `datatype`: `train`
 - `hide_labels`: `False`
 - `multitask_weights`: `[1]`
 - `batchsize`: `2`
 - `dynamic_batching`: `None`
 - `model`: `bart_classifier`
 - `dict_class`: `parlai.core.dict:DictionaryAgent`
 - `evaltask`: `message_history_orderhistorysincelastmovementphase_shortstate_pseudoorder_humanvsmodeldiscriminator_chunk`
 - `final_extra_opt`: ``
 - `eval_batchsize`: `None`
 - `eval_dynamic_batching`: `None`
 - `num_workers`: `8`
 - `display_examples`: `False`
 - `num_epochs`: `10.0`
 - `max_train_time`: `-1`
 - `max_train_steps`: `50000`
 - `early_stop_at_n_steps`: `-1`
 - `log_every_n_steps`: `100`
 - `validation_every_n_secs`: `-1`
 - `validation_every_n_steps`: `2000`
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
 - `log_every_n_secs`: `-1`
 - `distributed_world_size`: `64`
 - `ddp_backend`: `ddp`
 - `image_size`: `256`
 - `image_cropsize`: `224`
 - `model_generated_messages`: `denoising_justifications_extended_beam1`
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
 - `extend_order_history_since_last_n_movement_phase`: `2`
 - `extend_state_history_since_last_n_movement_phase`: `2`
 - `pseudo_order_generation`: `False`
 - `pseudo_order_generation_future_message`: `True`
 - `pseudo_order_generation_injected_sentence`: `None`
 - `pseudo_order_generation_inject_all`: `True`
 - `pseudo_order_generation_partner_view`: `False`
 - `pseudo_order_generation_current_phase_prefix`: `False`
 - `two_party_dialogue`: `False`
 - `no_speaker_dialogue_history`: `False`
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
 - `learningrate`: `2.6262121212121226e-05`
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
 - `warmup_updates`: `0`
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
 - `starttime`: `Jul29_07-59`
 - `rank`: `0`
</details>


## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).