#!/usr/bin/env python
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import subprocess
import tempfile

import heyhi

# Parse git status
lines = subprocess.check_output(["git", "status"]).strip().decode().split("\n")
lines = [l.strip() for l in lines]
lines = lines[(lines.index("Untracked files:") + 2) :]
if "" in lines:
    lines = lines[: lines.index("")]


# Append untracked files to blacklist
with open(heyhi.PROJ_ROOT / "pyrightconfig.CI.json") as f:
    j = json.load(f)
j["exclude"].extend(lines)

# Write new temp file and run pyright
name = f"pyrightconfig.local.json"
with open(name, "w") as f:
    json.dump(j, f)
try:
    subprocess.run(["pyright", "-p", name], check=True)
finally:
    subprocess.run(["rm", "-f", name], check=True)
