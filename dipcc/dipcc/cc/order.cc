/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <stdexcept>
#include <string>

#include "checks.h"
#include "loc.h"
#include "order.h"

namespace dipcc {

Order::Order(Unit unit, OrderType type, Unit target, Loc dest, bool via)
    : unit_(unit), type_(type), target_(target), dest_(dest), via_(via) {}

Order::Order(OwnedUnit unit, OrderType type, OwnedUnit target, Loc dest,
             bool via)
    : unit_(unit.unowned()), type_(type), target_(target.unowned()),
      dest_(dest), via_(via) {}

Order::Order(OwnedUnit unit, OrderType type, Unit target, Loc dest, bool via)
    : unit_(unit.unowned()), type_(type), target_(target), dest_(dest),
      via_(via) {}

Order::Order(Unit unit, OrderType type, Loc dest)
    : unit_(unit), type_(type), dest_(dest) {}
Order::Order(Unit unit, OrderType type, Loc dest, bool via)
    : unit_(unit), type_(type), dest_(dest), via_(via) {}

Loc loc_from_str_throws(const std::string &s) {
  Loc loc = loc_from_str(s);
  if (loc == Loc::NONE) {
    throw std::invalid_argument("Bad loc_from_str: " + s);
  }
  return loc;
}

void check_throws(bool b, const std::string &msg) {
  if (!b) {
    throw std::invalid_argument(msg);
  }
}

Order::Order(const std::string &s) {
  check_throws(s.size() >= 7, "Can't parse order: " + s);
  size_t i = 0;

  // Unit type
  check_throws(s[i] == 'A' || s[i] == 'F', "Can't parse order: " + s);
  unit_.type = s[i++] == 'A' ? UnitType::ARMY : UnitType::FLEET;
  check_throws(s[i++] == ' ', "Can't parse order: " + s);

  // Unit loc
  if (s[i + 3] == '/') {
    unit_.loc = loc_from_str_throws(s.substr(i, 6));
    i += 6;
  } else {
    unit_.loc = loc_from_str_throws(s.substr(i, 3));
    i += 3;
  }
  check_throws(s[i++] == ' ', "Can't parse order: " + s);

  // Order type
  char order_type = s[i++];

  if (order_type == 'H' || order_type == 'B' || order_type == 'D') {
    if (order_type == 'H') {
      type_ = OrderType::H;
    } else if (order_type == 'B') {
      type_ = OrderType::B;
    } else {
      type_ = OrderType::D;
    }
    check_throws(i == s.size(), "Can't parse order: " + s);
    return;
  }

  if (order_type == 'D') {
    // Disband
    type_ = OrderType::D;
    check_throws(i == s.size(), "Can't parse order: " + s);
    return;
  }

  check_throws(s[i++] == ' ', "Can't parse order: " + s);

  if (order_type == '-' || order_type == 'R') {
    // Move
    type_ = order_type == '-' ? OrderType::M : OrderType::R;

    // Move dest
    if (s[i + 3] == '/') {
      dest_ = loc_from_str_throws(s.substr(i, 6));
      i += 6;
    } else {
      dest_ = loc_from_str_throws(s.substr(i, 3));
      i += 3;
    }

    // maybe via?
    if (order_type == '-' && i != s.size()) {
      check_throws(i + 4 == s.size(), "Can't parse order: " + s);
      check_throws(s.substr(i) == " VIA", "Can't parse order: " + s);
      via_ = true;
    }
    return;
  }

  // Could be SM, SH, or C

  // Target unit
  check_throws(s[i] == 'A' || s[i] == 'F', "Can't parse order: " + s);
  target_.type = s[i++] == 'A' ? UnitType::ARMY : UnitType::FLEET;
  check_throws(s[i++] == ' ', "Can't parse order: " + s);

  // Target loc
  if (s[i + 3] == '/') {
    target_.loc = loc_from_str_throws(s.substr(i, 6));
    i += 6;
  } else {
    target_.loc = loc_from_str_throws(s.substr(i, 3));
    i += 3;
  }

  // Support hold - done parsing
  if (i == s.size()) {
    check_throws(order_type == 'S', "Can't parse order: " + s);
    type_ = OrderType::SH;
    return;
  }
  check_throws(s[i++] == ' ', "Can't parse order: " + s);
  check_throws(s[i++] == '-', "Can't parse order: " + s);
  check_throws(s[i++] == ' ', "Can't parse order: " + s);

  // Could be SM or C - parse dest
  if (s[i + 3] == '/') {
    dest_ = loc_from_str_throws(s.substr(i, 6));
    i += 6;
  } else {
    dest_ = loc_from_str_throws(s.substr(i, 3));
    i += 3;
  }

  // We should be done now
  check_throws(i == s.size(), "Can't parse order: " + s);
  if (order_type == 'C') {
    type_ = OrderType::C;
  } else if (order_type == 'S') {
    type_ = OrderType::SM;
  } else {
    check_throws(false, "Can't parse order: " + s);
  }
}

std::tuple<UnitType, Loc, OrderType, UnitType, Loc, Loc, bool>
Order::to_tuple() const {
  return std::tie(unit_.type, unit_.loc, type_, target_.type, target_.loc,
                  dest_, via_);
}

std::string Order::to_string() const {
  std::string s;
  s += unit_.to_string();

  switch (type_) {
  case OrderType::H: {
    s += " H";
    return s;
  }
  case OrderType::B: {
    s += " B";
    return s;
  }
  case OrderType::D: {
    s += " D";
    return s;
  }
  case OrderType::M: {
    s += " - ";
    s += loc_str(dest_);
    if (via_) {
      s += " VIA";
    }
    return s;
  }
  case OrderType::R: {
    s += " R ";
    s += loc_str(dest_);
    return s;
  }
  case OrderType::SH: {
    s += " S ";
    s += target_.to_string();
    return s;
  }
  case OrderType::SM: {
    s += " S ";
    s += target_.to_string();
    s += " - ";
    s += loc_str(dest_);
    return s;
  }
  case OrderType::C: {
    s += " C ";
    s += target_.to_string();
    s += " - ";
    s += loc_str(dest_);
    return s;
  }
  default: {
    JFAIL("Bad order type: " + std::to_string(static_cast<int>(type_)));
  }
  }
}

bool Order::operator==(const Order &other) const {
  return this->get_unit() == other.get_unit() &&
         this->get_type() == other.get_type() &&
         this->get_target() == other.get_target() &&
         this->get_dest() == other.get_dest() &&
         this->get_via() == other.get_via();
}

std::ostream &operator<<(std::ostream &os, const Order &x) {
  return os << "Order(\"" << x.to_string() << "\")";
}

Order Order::with_via(bool via) const {
  return Order(this->get_unit(), this->get_type(), this->get_target(),
               this->get_dest(), via);
}

// Comparator function implemening LOCS-ordering
bool loc_order_cmp(const Order &a, const Order &b) {
  Loc loc_a = a.get_unit().loc;
  Loc loc_b = b.get_unit().loc;
  JCHECK(loc_a != Loc::NONE, "loc_order_cmp called with Loc::NONE");
  JCHECK(loc_b != Loc::NONE, "loc_order_cmp called with Loc::NONE");
  return LOC_IDX.at(loc_a) < LOC_IDX.at(loc_b);
}

Order Order::as_normalized() const {
  Order order(*this);
  order.target_.loc = root_loc(order.target_.loc);
  return order;
}

} // namespace dipcc
