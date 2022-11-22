#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import os
import pickle
import time
from collections import defaultdict
from pprint import pformat

from names import FULL_TO_SHORT

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

IGNORE_TESTS = ["6.B.11."]
OVERRIDE_OWNERS = {"6.A.6.": {"LON": "ENGLAND"}, "6.B.13.": {"BUL/SC": "RUSSIA"}}
OVERRIDE_ORDERS = {
    "6.B.13.": {"RUSSIA": {"BUL/SC": "F BUL/SC - CON"}},
    "6.D.34.": {"ITALY": {"PRU": "A PRU S A LVN - PRU"}},
}


class IgnoreTest(Exception):
    pass


def parse_test(html):
    test_name = html.lstrip("<h4>").strip().split(" ", 1)[0]
    if test_name in IGNORE_TESTS:
        raise IgnoreTest()
    test_description = html.lstrip("<h4>").split("</h4>", 1)[0]
    splits = re.compile("</?pre>").split(html)
    assert len(splits) == 3, "Bad: " + test_name
    orders = parse_orders(splits[1], override_orders=OVERRIDE_ORDERS.get(test_name, {}))
    result_str = splits[2].strip().replace("\n", " ")
    return test_name, test_description, orders, result_str


def parse_orders(s, override_orders={}):
    r = defaultdict(list)
    current_power = None
    for line in s.strip().upper().split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        for p in POWERS:
            if line.startswith(p):
                current_power = p
                break
        else:
            if any(line.startswith(pre) for pre in ["A ", "F ", "BUILD ", "DISBAND ", "REMOVE "]):
                # parse line
                for full, short in FULL_TO_SHORT.items():
                    if full in line:
                        line = line.replace(full, short)
                for c in ["NC", "EC", "SC", "WC"]:
                    line = line.replace(f"({c})", f"/{c}")
                for a, b in [
                    ("SUPPORTS", "S"),
                    ("CONVOYS", "C"),
                    ("BUILD", "B"),
                    ("DISBAND", "D"),
                    ("REMOVE", "D"),
                    ("HOLD", "H"),
                    ("VIA CONVOY", "VIA"),
                ]:
                    line = line.replace(a, b)

                # lines are sometimes written "B F STP/NC" instead of "F STP/NC B"
                for pre in ["B ", "D "]:
                    if line.startswith(pre):
                        line = line[2:] + " " + pre[0]

                line = override_orders.get(current_power, {}).pop(line.split()[1], line)
                r[current_power].append(line)
            else:
                raise Exception("Bad line: " + line)

    # handle remaining override orders
    for power, d in override_orders.items():
        for order in d.values():
            r[power].append(order)

    return dict(r)


def parse_tests_from_html(html_path):
    with open(html_path, "r") as f:
        html = f.read()
    splits = re.compile("""a name="[6]\.[A-Z]\.[0-9]+">""").split(html)
    test_strs = splits[1:-1]
    parsed_tests = []
    for s in test_strs:
        try:
            parsed_tests.append(parse_test(s))
        except IgnoreTest:
            continue
        except AssertionError:
            continue
    return parsed_tests


if __name__ == "__main__":
    # read input cache
    CACHE_FILE = "parse_datc_cache.pkl"
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            user_inputs = pickle.load(f)
    else:
        user_inputs = {}

    # read html file and extract orders
    parsed_tests = parse_tests_from_html("datc.html")

    stats_t, stats_count = 0, 0
    try:
        for i, (test_name, test_description, orders, result_str) in enumerate(parsed_tests):
            if test_name in user_inputs:
                continue
            print(chr(27) + "[2J")  # clear screen
            print(f"{i} / {len(test_strs)}\tavg={stats_t/(stats_count+.0001)}")
            print(s, pformat(orders), result_str, sep="\n\n")
            t_start = time.time()
            user_input = input(
                "\nWhich locs' moves should succeed? [NO(NE), ALL, !LOC, LOC H, LOC D, SKIP]: "
            )
            t_elapsed = time.time() - t_start
            user_inputs[test_name] = user_input

            # some stats keeping
            stats_t += t_elapsed
            stats_count += 1

    finally:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(user_inputs, f)
