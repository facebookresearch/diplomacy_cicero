# base_strategy_model_0_5_policy.ckpt model card

## Overview

This model is an unpublished model that uses a transformer encoder, trained to imitate human play via behavioral cloning in Diplomacy. This model was also jointly trained to predict final game scores. It is used only as the policy of a benchmark opponent for statistics during RL training.

This model was developed by Meta AI. It was trained January 2021.

### Architecture details

This model has roughly 8 million trainable parameters. Its architecture contains some but not all of the developments in model architecture and feature encoding between the publications of https://arxiv.org/abs/2010.02923 and https://arxiv.org/abs/2210.05492. Since it is an otherwise unpublished model, there isn't a detailed reference for this model, ultimately see the metadata in the model file itself and/or the source code.

## Intended Use

This model is for research purposes only.

## Limitations

This model does not condition on or handle dialogue. It also decodes only one player's action at a time, predicting its probability independently from that of any other player.

## Datasets used

This model was trained using the webDiplomacy dataset, described in the [paper](https://arxiv.org/abs/2010.02923).

## Privacy

This model does not observe any personally identifiable information. It was trained to predict actions and/or values in Diplomacy games on an anonymized dataset of such games.

## Hyperparameters
<details>
<summary> Hyperparameters </summary>

 - `val_set_pct`: `0.01`
 - `lstm_dropout`: `0.2`
 - `encoder_dropout`: `0.2`
 - `learnable_A`: `False`
 - `learnable_alignments`: `False`
 - `avg_embedding`: `False`
 - `num_encoder_blocks`: `10`
 - `value_loss_weight`: `0.7`
 - `value_decoder_init_scale`: `0.01`
 - `value_decoder_clip_grad_norm`: `1e-07`
 - `min_rating_percentile`: `0.5`
 - `lstm_layers`: `2`
 - `featurize_output`: `True`
 - `relfeat_output`: `True`
 - `residual_linear`: `True`
 - `merged_gnn`: `True`
 - `value_softmax`: `True`
 - `use_global_pooling`: `False`
 - `inter_emb_size`: `112`
 - `encoder`: `{'transformer': {'num_blocks': 10, 'dropout': 0.2, 'layerdrop': 0.0, 'num_heads': 8, 'ff_channels': 192}}`
</details>



## Feedback

We would love any feedback about the model. Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/diplomacy_cicero/issues).
