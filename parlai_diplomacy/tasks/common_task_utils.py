#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


###########################################
# CONSTANTS
###########################################

REDACTED_TOKEN = "[REDACTED]"

# Discriminative classifier and tasks constants
REAL = "REAL"
CORRUPTED = "CORRUPTED"
INCORRECT_RECIPIENT = "INCORRECT_RECIPIENT"
CORRUPTED_ENTITY = "CORRUPTED_ENTITY"
INCORRECT_PHASE = "INCORRECT_PHASE"
INCORRECT_GAME = "INCORRECT_GAME"
REAPEATED_MESSAGE = "REAPEATED_MESSAGE"

# Lie detector classifier
LIE_TOKEN = "LIE"
NOT_LIE_TOKEN = "NOT_LIE"

CHATTINESS_BUCKETS = list(range(0, 100 + 1, 5))
EPSILON = 1e-10
