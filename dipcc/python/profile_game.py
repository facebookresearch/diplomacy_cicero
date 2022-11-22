#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import json
import time
from pprint import pprint

from fairdiplomacy import pydipcc

parser = argparse.ArgumentParser()
parser.add_argument("game_json")
args = parser.parse_args()

with open(args.game_json) as f:
    json_s = f.read()
j = json.loads(json_s)


cc_game = pydipcc.Game()


times = {"cc": 0.0}
n = 0

for phase_data in j["phases"]:
    phase = phase_data["name"]
    print(phase)

    assert cc_game.get_state()["name"] == phase, cc_game.get_state()["name"]

    if "orders" not in phase_data or phase == "COMPLETED":
        break

    all_orders = phase_data["orders"]
    pprint(phase)
    pprint(all_orders)

    for game, t in [(cc_game, "cc")]:
        for power, orders in all_orders.items():
            if orders:
                game.set_orders(power, orders)

        t_start = time.time()
        game.process()
        times[t] += time.time() - t_start

    n += 1

print("\n## Process")
print(f"cc: {times['cc']/n*1e3:.3f}ms/phase")
print(f"py: {times['py']/n*1e3:.3f}ms/phase")
print(f"py / cc = {times['py']/times['cc']:.0f}")
