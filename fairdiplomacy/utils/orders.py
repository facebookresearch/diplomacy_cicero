#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from fairdiplomacy.typedefs import Location, Order


def is_hold(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 3 and pieces[2] == "H"


def is_move(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) >= 4 and pieces[2] == "-"


def is_move_with_via(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 5 and pieces[2] == "-" and pieces[4] == "VIA"


def is_support(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) >= 5 and pieces[2] == "S"


def is_support_hold(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 5 and pieces[2] == "S"


def is_support_move(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 7 and pieces[2] == "S" and pieces[5] == "-"


def is_convoy(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 7 and pieces[2] == "C" and pieces[5] == "-"


def is_retreat(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 4 and pieces[2] == "R"


def is_disband(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 3 and pieces[2] == "D"


def is_build(order: Order) -> bool:
    pieces = order.split()
    return len(pieces) == 3 and pieces[2] == "B"


def get_unit_type(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 2
    return pieces[0]


def get_unit_location(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 2
    return pieces[1]


def get_move_or_retreat_destination(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 4 and (pieces[2] == "-" or pieces[2] == "R")
    return pieces[3]


def get_supported_unit_type(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 5 and pieces[2] == "S"
    return pieces[3]


def get_supported_unit_location(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 5 and pieces[2] == "S"
    return pieces[4]


def get_supported_unit_destination(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) == 7 and pieces[2] == "S" and pieces[5] == "-"
    return pieces[6]


def get_convoyed_unit_location(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) >= 5 and pieces[2] == "C"
    return pieces[4]


def get_convoyed_unit_destination(order: Order) -> Location:
    pieces = order.split()
    assert len(pieces) == 7 and pieces[2] == "C" and pieces[5] == "-"
    return pieces[6]
