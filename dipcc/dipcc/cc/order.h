/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <glog/logging.h>
#include <tuple>

#include "checks.h"
#include "enums.h"
#include "loc.h"
#include "owned_unit.h"
#include "unit.h"

namespace dipcc {

class Order {
public:
  // Constructors
  Order() {}
  Order(Unit unit, OrderType type, Unit target = {}, Loc dest = Loc::NONE,
        bool via = false);
  Order(OwnedUnit unit, OrderType type, OwnedUnit target = {},
        Loc dest = Loc::NONE, bool via = false);
  Order(OwnedUnit unit, OrderType type, Unit target = {}, Loc dest = Loc::NONE,
        bool via = false);
  Order(Unit unit, OrderType type, Loc dest);
  Order(Unit unit, OrderType type, Loc dest, bool via);
  Order(const std::string &s);

  // Getters
  Unit get_unit() const { return unit_; }
  OrderType get_type() const { return type_; }
  Unit get_target() const { return target_; }
  Loc get_dest() const { return dest_; }
  bool get_via() const { return via_; }

  // Convert to order string
  std::string to_string() const;

  // Return a copy of this order with via set explicitly
  Order with_via(bool via) const;

  // Return a copy with vague coastal variants where possible
  Order as_normalized() const;

  // Comparator (to enable use as set/map key)
  std::tuple<UnitType, Loc, OrderType, UnitType, Loc, Loc, bool>
  to_tuple() const;
  bool operator<(const Order &other) const {
    return this->to_tuple() < other.to_tuple();
  }

  // Equality comparator
  bool operator==(const Order &other) const;

  // Print operator
  friend std::ostream &operator<<(std::ostream &os, const Order &);

private:
  // Members
  Unit unit_;
  OrderType type_;
  Unit target_ = {UnitType::NONE, Loc::NONE}; // Used for SH, SM, C
  Loc dest_ = Loc::NONE;                      // Used for M, SM, C
  bool via_ = false;                          // Used for M
};

// Comparator function implemening LOCS-ordering
bool loc_order_cmp(const Order &a, const Order &b);

} // namespace dipcc
