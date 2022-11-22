#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Datapath constants.

NOTE: NONE OF THE PATHS HERE WILL WORK! In order to use this code, you will have to process all data and annotations separately and add the paths here.
"""
import os

######################################################
#  Task Version information
######################################################
LATEST_VERSION = 1


######################################################
#  Game data
######################################################
LATEST_DATA_DIR = "<PATH TO DATA DIRECTORY>"
# all game jsons
GAME_JSONS = os.path.join(LATEST_DATA_DIR, "all_games/game_*.json*")
# full press only
FULLPRESS_GAME_JSONS = os.path.join(LATEST_DATA_DIR, "full_press_games/game_*.json*")


######################################################
#  Metadata & extra annotations
######################################################
GAME_METADATA_PATH = os.path.join(LATEST_DATA_DIR, "metadata.json")
CHATTINESS_METADATA_PATH = None  # DEPRECATED

# Pseudo orders
PSEUDO_ORDER_SINGLETURN_DIR = None
PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR = os.path.join(
    LATEST_DATA_DIR, "extra_annotations/pseudo_orders_20220428"
)
PSEUDO_ORDER_SINGLEVIEW_SINGLETURN_DIR_VERSION = 1  # Computed on version 1
PSEUDO_ORDER_PREFIX_ROLLOUT_DIR = os.path.join(
    LATEST_DATA_DIR, "extra_annotations/prefix_rollout_pseudo_orders_20220429"
)
PSEUDO_ORDER_PREFIX_ROLLOUT_DIR_VERSION = 3  # Computed on version 3


# Generated messages
NUCLEUS_0_9_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/bart_rollout_singleview_pseudos_20210916_dialogue_generation_20211107",
)

NUCLEUS_0_5_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_dialogue_generated_using_edinan_20210916_rollout_singleview_pseudos_87d_model_topp0.5_20211224",
)

NUCLEUS_0_99_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_dialogue_generated_using_edinan_20210916_rollout_singleview_pseudos_87d_model_topp0.99_20211222",
)

SHORTCONTEXT_NUCLEUS_0_9_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_dialogue_generated_using_edinan_20211103_shortcontext_bartdialogue_c18_model_20211224",
)
SHORTCONTEXT_NUCLEUS_0_5_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_dialogue_generated_using_edinan_20211103_shortcontext_bartdialogue_c18_model_nucleus_0.5_20220104",
)
SHORTCONTEXT_NUCLEUS_0_99_GENERATED_MESSAGES_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_dialogue_generated_using_edinan_20211103_shortcontext_bartdialogue_c18_model_nucleus_0.99_20220104",
)

DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED1_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_denoised_dialogue_generated_using_mikelewis_20220422_denoising_9c1_model_mask_seed1_20220614",
)

DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED2_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_denoised_dialogue_generated_using_mikelewis_20220422_denoising_9c1_model_mask_seed2_20220614",
)

DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED3_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_denoised_dialogue_generated_using_mikelewis_20220422_denoising_9c1_model_mask_seed3_20220613",
)

DENOISING_NOISY_LOCATIONS_ONLY_NUCLEUS_0_9_GENERATED_MESSAGES_SEED1_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_denoised_dialogue_generated_using_denoising_targeted_noisy_locations_denoising_single_token_masking_only_cb7_model_targeted_cardinals_denoising_single_token_masking_only_seed1_20220624",
)

DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_ALL_SEED_DIRS = [
    DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED1_DIR,
    DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED2_DIR,
    DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED3_DIR,
]

DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220720_BEAM0_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_justification_generation_weakmodel_20220720_beam0",
)

DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220720_BEAM1_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_justification_generation_weakmodel_20220720_beam1",
)

DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220727_BEAM0_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_justification_generation_weakmodel_20220727_beam0",
)

DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220727_BEAM1_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_justification_generation_weakmodel_20220727_beam1",
)

DENOISING_CARDINALS_SEED1_DIR = os.path.join(
    LATEST_DATA_DIR, "extra_annotations/model_dialogues_denoised_cardinal_examples_seed1_20220729",
)

DENOISING_NEGATIONS_SEED1_DIR = os.path.join(
    LATEST_DATA_DIR,
    "extra_annotations/model_dialogues_denoised_dialogue_generated_using_negations_ATTEMPT1_seed1212121212_20220729",
)

GENERATED_MESSAGES_DIRS = {
    "nucleus_0.9": [NUCLEUS_0_9_GENERATED_MESSAGES_DIR],
    "nucleus_0.5": [NUCLEUS_0_5_GENERATED_MESSAGES_DIR],
    "nucleus_0.99": [NUCLEUS_0_99_GENERATED_MESSAGES_DIR],
    "shortcontext": [SHORTCONTEXT_NUCLEUS_0_9_GENERATED_MESSAGES_DIR],
    "shortcontext_0.5": [SHORTCONTEXT_NUCLEUS_0_5_GENERATED_MESSAGES_DIR],
    "shortcontext_0.99": [SHORTCONTEXT_NUCLEUS_0_99_GENERATED_MESSAGES_DIR],
    "denoising": DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_ALL_SEED_DIRS,
    "denoising_singleseed": [DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED1_DIR],
    "denoising_noisy_locations": [
        DENOISING_NOISY_LOCATIONS_ONLY_NUCLEUS_0_9_GENERATED_MESSAGES_SEED1_DIR
    ],
    "denoising_singleseed_seed2": [DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED2_DIR],
    "denoising_singleseed_seed3": [DENOISING_NUCLEUS_0_9_GENERATED_MESSAGES_SEED3_DIR],
    "denoising_cardinals": [DENOISING_CARDINALS_SEED1_DIR],
    "denoising_justifications": [DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220720_BEAM0_DIR],
    "denoising_justifications_beam1": [
        DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220720_BEAM1_DIR
    ],
    "denoising_justifications_extended": [
        DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220727_BEAM0_DIR
    ],
    "denoising_justifications_extended_beam1": [
        DENOSING_JUSTIFICATIONS_GENERATED_MESSAGES_20220727_BEAM1_DIR
    ],
    "denoising_negations": [DENOISING_NEGATIONS_SEED1_DIR],
}
######################################################
# Discriminator teacher data
######################################################
DISCRIMINATOR_VALID_DATA_ROOT = "<ROOT TO DISCRIMINATOR DATA>"
DISCRIMINATOR_VALID_TEST_DATA = "data/valid_teacher_data.jsonl"
DISCRIMINATOR_SEQ_VALID_DATA_ROOT = "<PATH TO VALIDATION DATA FOR DISCRIMINATOR AGENT>"

#######################################################
# Validation / Test Game IDs
#######################################################
TEST_ID_PATH = LATEST_DATA_DIR + "test_set_ids.txt"
VALID_GAME_IDS = [
    120754,
    120790,
    120804,
    120835,
    120845,
    120860,
    120867,
    120873,
    120890,
    120916,
    120920,
    120922,
    120923,
    120924,
    120926,
    120927,
    120931,
    120936,
    120939,
    120943,
    120946,
    120948,
    120950,
    120954,
    120957,
    120959,
    120962,
    120964,
    120965,
    120968,
    120970,
    120971,
    120973,
    120976,
    120977,
    120978,
    120980,
    120981,
    120984,
    120987,
    120994,
    120996,
    120998,
    120999,
    121001,
    121004,
    121005,
    121007,
    121009,
    121017,
    121018,
    121019,
    121020,
    121025,
    121027,
    121033,
    121035,
    121037,
    121039,
    121043,
    121046,
    121047,
    121048,
    121050,
    121053,
    121054,
    121058,
    121061,
    121065,
    121073,
    121075,
    121079,
    121080,
    121081,
    121082,
    121084,
    121086,
    121087,
    121088,
    121095,
    121096,
    121099,
    121102,
    121105,
    121106,
    121107,
    121108,
    121110,
    121115,
    121117,
    121130,
    121132,
    121145,
    121153,
    121159,
    121161,
    121162,
    121165,
    121170,
    121206,
]


#######################################################
# Games with draw votes
#######################################################
# Fle with train games with draw votes
DRAW_VOTE_TRAIN_GAMES_FLE = "<PATH TO TEXT FILE CONTAINING A LIST OF TRAIN GAMES WITH DRAW VOTES>"
# Test IDs with draw votes
DRAW_TEST_IDS = [488121, 488372, 488347, 488037, 488219]
