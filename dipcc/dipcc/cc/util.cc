/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <glog/logging.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hash.h"
#include "loc.h"
#include "order.h"
#include "util.h"

namespace dipcc {

bool is_implicit_via(const Order &order,
                     const std::set<Order> &loc_possible_orders) {
  return order.get_type() == OrderType::M && !order.get_via() &&
         // order not possible
         !set_contains(loc_possible_orders, order) &&
         // order with via is possible
         set_contains(loc_possible_orders, order.with_via(true));
}

bool is_implicit_via(
    const Order &order,
    const std::unordered_map<Loc, std::set<Order>> &all_possible_orders) {
  auto it = all_possible_orders.find(order.get_unit().loc);
  return it != all_possible_orders.end() && is_implicit_via(order, it->second);
}

} // namespace dipcc
