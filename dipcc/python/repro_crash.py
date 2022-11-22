#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../dipcc/python/")
import pydipcc

parser = argparse.ArgumentParser()
parser.add_argument("crash_dump")
parser.add_argument("orders_json", nargs="?")
parser.add_argument("--phase", "-p", help="Which phase to replay? e.g. F1905M")
args = parser.parse_args()

with open(args.crash_dump, "r") as f:
    j = json.load(f)
game = pydipcc.Game.from_json(json.dumps(j))

if args.orders_json:
    print("Using orders from file:", args.orders_json)
    with open(args.orders_json, "r") as f:
        all_orders = json.load(f)
elif "staged_orders" in j:
    print("Using staged_orders")
    all_orders = j["staged_orders"]
elif args.phase:
    print("Replaying phase", args.phase)
    all_orders = next(p for p in game.get_phase_history() if p.name == args.phase).orders
    game = game.rolled_back_to_phase_start(args.phase)
else:
    # use last orders
    print("Using last phase's orders")
    last_phase = game.get_phase_history()[-1]
    all_orders = last_phase.orders
    game = game.rolled_back_to_phase_start(last_phase.name)


# Replay the phase
for power, orders in all_orders.items():
    game.set_orders(power, orders)

game.process()

print("SUCCESS!")
print("Next phase:", game.phase)
