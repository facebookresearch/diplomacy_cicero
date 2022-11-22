#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Small method to clean metadata string
from typing import Any, List, Optional, Tuple


def clean_token_metadata(token_details: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    # replace non-ascii characters
    new = [
        (tok.replace("\\xc4", "").replace("\\xa0", " ").replace("\\x8a", "\n"), metadata)
        for tok, metadata in token_details
    ]

    # remove prefix
    get_colon_index = next(i for i, (tok, _) in enumerate(new) if ":" in tok)
    new = [details for i, details in enumerate(new) if i > get_colon_index]

    # remove extra spaces from tokens as well as end-of-message token
    new = [
        (tok.strip(), {**metadata, "token_logprob": str(metadata["token_logprob"])})
        for (tok, metadata) in new
        if tok not in ["", " "]
    ]

    return new
