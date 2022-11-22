# ParlAI Diplomacy Utils

This subdirectory contains useful utilities related to training and evaluating Transformer-based Diplomacy models trained in ParlAI.

## Utility Files and Folders

- **game2seq** : most importantly -- this folder contains *all* utilities for converting game JSONs into the strings expected as input for the Transformers and for converting the string output of Transformers back into a structured format
    - **format_helpers**: folder containing helper functions for formatting specific pieces of input or output, such as messages, orders, or state
    - **base_prediction**: base formatting class which takes game objects as input and returns formatted strings for training or inference
    - **dialogue_prediction**: formatting class for dialogue prediction tasks
    - **factory**: sequence formatter factory, which determines which sequences formatter should be used based on the task string; e.g. `message_history_state_dialogue_chunk` routes to the dialogue prediction formatter
    - **input_validation**: system of tests for validating that the input formatting is as expected for a model based on the arguments
    - **order_prediction**: formatting class for order predictin classes
    - **typing**: diplomacy specific type definitions
- **datapath_constants**: file containing a list of paths to data and annotation
- **loading**: task and agent registry -- for use of `diplom` supercommand
- **misc**: miscellaneous utilities
- **nucleus_scoring_helpers**: helper functions for scoring a sequence based on whether or not it is contained in the "nucleus"
- **pseudo_orders**: utilities for loading train-time pseudo order annotations
- **special_tokens**: utilities related to Diplomacy-specific special tokens
- **token_metadata**: generation-time token metadata formatting
- **webdip_games**: utilities related to parsing text from webdip games