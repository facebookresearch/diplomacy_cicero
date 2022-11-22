#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import pickle

from parse_datc import parse_tests_from_html, OVERRIDE_OWNERS

HTML_PATH = os.path.join(os.path.dirname(__file__), "datc.html")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "parse_datc_cache.pkl")

DISABLED_TESTS = {"6.H.11."}


def render_and_print_test(name, description, orders, result_raw):
    safe_name = name.replace(".", "")
    maybe_disabled = "DISABLED_" if name in DISABLED_TESTS else ""
    orders_by_loc = {
        order.split()[1]: order for power, porders in orders.items() for order in porders
    }
    print("//", description.strip())
    print(f"TEST_F(DATCTest, {maybe_disabled}Test{safe_name}) {{")
    print(f"GameState state;")
    for power, power_orders in orders.items():
        for order in power_orders:
            unit_type, loc, *rest = order.split()
            unit_type_full = {"A": "ARMY", "F": "FLEET"}[unit_type]
            loc_safe = loc.replace("/", "_")
            owner = OVERRIDE_OWNERS.get(name, {}).get(loc, power)
            print(f"state.set_unit(Power::{owner}, UnitType::{unit_type_full}, Loc::{loc_safe});")
    print()

    # print all possible orders
    print("for (auto it_loc_orders : state.get_all_possible_orders()) {")
    print('LOG(INFO) << "ALL_POSSIBLE_ORDERS: " << loc_str(it_loc_orders.first);')
    print('for (auto &order : it_loc_orders.second) { LOG(INFO) << "  " << order.to_string(); } }')

    print("std::unordered_map<Power, std::vector<Order>> orders;")
    for power, power_orders in orders.items():
        for order in power_orders:
            print(f"""orders[Power::{power}].push_back(Order("{order}"));""")
    print()
    print("GameState next(state.process(orders));")
    print()
    print("auto next_possible_orders(next.get_all_possible_orders());")
    print("set<Loc> next_orderable_locs;")
    print("for (auto& it : next.get_orderable_locations()) {")
    print("  for (Loc loc : it.second) { next_orderable_locs.insert(loc); }")
    print("}")
    print()
    if result_raw in ["NONE", "NO", "N"]:
        print("// expect no unit moves")
        print("EXPECT_THAT(next.get_units(), testing::ContainerEq(state.get_units()));")
    elif result_raw in ["ALL"]:
        for order in orders_by_loc.values():
            if order.split()[2] == "-":
                render_and_print_expect_success(order)
    else:
        raw_constraints = [s.strip() for s in result_raw.split(",")]
        for raw in raw_constraints:
            if raw.startswith("*"):
                order = raw[1:].strip('"')
                render_and_print_expect_retreat_order(order, possible=True)
            elif raw.startswith("!"):
                raw = raw[1:].strip('"')
                if " " in raw:
                    render_and_print_expect_retreat_order(raw, possible=False)
                else:
                    order = orders_by_loc[raw]
                    render_and_print_expect_success(order, success=False)
            elif raw.endswith(" D"):
                render_and_print_expect_dislodge(raw.split()[0], dislodge=True)
            elif raw.endswith(" H"):
                render_and_print_expect_dislodge(raw.split()[0], dislodge=False)
            elif raw.endswith(" X"):
                loc = raw.split()[0]
                order = orders_by_loc[loc]
                render_and_print_expect_success(order, success=False)
                render_and_print_expect_dislodge(loc, dislodge=True)
            else:
                order = orders_by_loc[raw]
                render_and_print_expect_success(order, success=True)

    print("}\n")


def render_and_print_expect_move(src, dest, *, success=True):
    TF = "TRUE" if success else "FALSE"
    print(
        f"EXPECT_{TF}(state.get_unit(Loc::{src}).power == next.get_unit(Loc::{dest}).power && state.get_unit(Loc::{src}).type == next.get_unit(Loc::{dest}).type);"
    )


def render_and_print_expect_success(order, *, success=True):
    if order.split()[2] == "S":
        order = " ".join(order.split()[3:])
    if order.split()[2] == "H":
        render_and_print_expect_dislodge(order.split()[1], dislodge=not success)
    elif order.split()[2] == "B":
        assert False, "hi mom"
    else:
        assert order.split()[2] == "-", "Not a move or support-move order: " + order
        dest_safe = order.split()[3].replace("/", "_")
        loc_safe = order.split()[1].replace("/", "_")
        print(f"// expect {'success' if success else 'failure'}: {order}")
        render_and_print_expect_move(loc_safe, dest_safe, success=success)


def render_and_print_expect_dislodge(loc, *, dislodge=True):
    loc = loc.replace("/", "_")
    print(f"// expect {loc}{'' if dislodge else ' not'} dislodged")
    if dislodge:
        # Don't expect a retreat order: unit may be force-disbanded. Only
        # expect unit to be gone.
        render_and_print_expect_move(loc, loc, success=False)
    else:
        render_and_print_expect_move(loc, loc, success=True)
        print("if (next.get_phase().phase_type == 'R') {")
        print(f"EXPECT_THAT(next_orderable_locs, testing::Not(testing::Contains(Loc::{loc})));")
        print("}")


def render_and_print_expect_retreat_order(order, *, possible=True):
    loc = order.split()[1].replace("/", "_")
    maybe_not = "" if possible else "not "
    print(f"// expect retreat order {maybe_not}possible: {order}")
    if possible:
        print(f"EXPECT_EQ(next.get_phase().phase_type, 'R') << \"Should be retreat phase\";")
        print(
            f'EXPECT_THAT(next_possible_orders[Loc::{loc}], testing::Contains(Order("{order}")));'
        )
    else:
        print("if (next.get_phase().phase_type == 'R') {")
        print(
            f'EXPECT_THAT(next_possible_orders[Loc::{loc}], testing::Not(testing::Contains(Order("{order}"))));'
        )
        print("}")


HEADER = """
// This file is auto-generated by render_tests.py
//
#include <vector>
#include "../cc/game.h"
#include "../cc/hash.h"
#include "../cc/thirdparty/nlohmann/json.hpp"
#include "consts.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;

namespace dipcc {

class DATCTest : public ::testing::Test {};
"""

FOOTER = """} // namespace dipcc"""

if __name__ == "__main__":
    parsed_tests = parse_tests_from_html(HTML_PATH)

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    print(HEADER)

    for test_name, test_description, orders, result_str in parsed_tests:
        result_raw = cache.get(test_name, "SKIP").upper().strip()
        if result_raw == "SKIP":
            continue

        if any(order.split()[2] not in "HCS-" for porders in orders.values() for order in porders):
            continue
        if any(s in test_name for s in ["6.G."]):
            continue

        try:
            render_and_print_test(test_name, test_description, orders, result_raw)
        except Exception as e:
            print("Caught during render_and_print_test:", test_name, file=sys.stderr)
            raise e

    print(FOOTER)
