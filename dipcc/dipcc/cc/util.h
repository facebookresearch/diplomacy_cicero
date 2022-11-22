/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loc.h"
#include "order.h"

#include "thirdparty/nlohmann/json.hpp"

namespace dipcc {

bool is_implicit_via(const Order &order,
                     const std::set<Order> &loc_possible_orders);

bool is_implicit_via(
    const Order &order,
    const std::unordered_map<Loc, std::set<Order>> &all_possible_orders);

template <typename T> bool set_contains(const std::set<T> &c, const T &x) {
  return c.find(x) != c.end();
}

template <typename T>
bool set_contains(const std::unordered_set<T> &c, const T &x) {
  return c.find(x) != c.end();
}

template <typename T, typename H>
bool set_contains(const std::unordered_set<T, H> &c, const T &x) {
  return c.find(x) != c.end();
}

template <typename T, typename Q>
bool map_contains(const std::map<T, Q> &c, const T &x) {
  return c.find(x) != c.end();
}

template <typename T, typename Q>
bool map_contains(const std::unordered_map<T, Q> &c, const T &x) {
  return c.find(x) != c.end();
}

template <typename T> bool vec_contains(const std::vector<T> &c, const T &x) {
  return std::find(c.begin(), c.end(), x) != c.end();
}

template <typename K, typename V>
bool safe_contains(const std::unordered_map<K, std::vector<V>> &c, const K &k,
                   const V &v) {
  auto it = c.find(k);
  return it != c.end() && vec_contains(it->second, v);
}

template <typename K, typename V>
bool safe_contains(const std::unordered_map<K, std::set<V>> &c, const K &k,
                   const V &v) {
  auto it = c.find(k);
  return it != c.end() && set_contains(it->second, v);
}

template <typename... S, typename D>
D json_deep_get(const json &j, D default_return, const std::string &key) {
  auto it = j.find(key);
  if (it == j.end()) {
    return default_return;
  } else {
    return j[key];
  }
}

template <typename... S, typename D>
D json_deep_get(const json &j, D default_return, const std::string &key,
                S... keys) {
  auto it = j.find(key);
  if (it == j.end()) {
    return default_return;
  }

  if (sizeof...(keys) == 0) {
    return j[key];
  } else {
    return json_deep_get(j[key], default_return, keys...);
  }
}

} // namespace dipcc
