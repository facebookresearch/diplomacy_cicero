/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "enums.h"
#include "loc.h"
#include "order.h"
#include "owned_unit.h"
#include "phase.h"
#include "unit.h"

namespace std {

// template <> struct hash<dipcc::Order> {
//   size_t operator()(const dipcc::Order &order) const {
//     std::size_t r = 0;
//     // hash_combine(r, order.get_unit().type);
//     // hash_combine(r, order.get_unit().loc);
//     // hash_combine(r, order.get_type());
//     // hash_combine(r, order.get_target().type);
//     // hash_combine(r, order.get_target().loc);
//     // hash_combine(r, order.get_dest());
//     // hash_combine(r, order.get_via());
//     return r;
//   }
// };

} // namespace std

namespace dipcc {

template <class T> inline void hash_combine(std::size_t &s, const T &v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

template <> inline void hash_combine(std::size_t &seed, const Unit &unit) {
  hash_combine(seed, unit.type);
  hash_combine(seed, unit.loc);
}

template <> inline void hash_combine(std::size_t &seed, const OwnedUnit &unit) {
  hash_combine(seed, unit.power);
  hash_combine(seed, unit.type);
  hash_combine(seed, unit.loc);
}

template <> inline void hash_combine(std::size_t &seed, const Phase &phase) {
  hash_combine(seed, phase.season);
  hash_combine(seed, phase.phase_type);
  hash_combine(seed, phase.year);
}

struct HashOrder {
  std::size_t operator()(const Order &x) const {
    std::size_t r = 0;
    hash_combine(r, x.get_unit());
    hash_combine(r, x.get_type());
    hash_combine(r, x.get_target());
    hash_combine(r, x.get_dest());
    hash_combine(r, x.get_via());
    return r;
  }
};

} // namespace dipcc
