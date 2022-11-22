/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace nest {
// Magic from https://en.cppreference.com/w/cpp/utility/variant/visit
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;

template <typename T>
struct Nest {
  using leaf_type = T;
  using value_t =
      std::variant<T, std::vector<Nest>, std::map<std::string, Nest>>;
  Nest(std::vector<T> entries) : value(std::vector<Nest>()) {
    auto &v = std::get<std::vector<Nest>>(value);
    v.reserve(entries.size());
    for (auto &e : entries) {
      v.emplace_back(std::move(e));
    }
  }
  Nest(std::map<std::string, T> entries)
      : value(std::map<std::string, Nest>()) {
    auto &m = std::get<std::map<std::string, Nest>>(value);
    for (auto &p : entries) {
      m.emplace_hint(m.end(), std::move(p.first), std::move(p.second));
    }
  }
  Nest() = default;  // needed for type_caster below.
  Nest(const Nest &) = default;
  Nest(Nest &&) = default;
  Nest &operator=(const Nest &) = default;

  Nest(value_t v) : value(std::move(v)) {}

  value_t value;

  bool is_leaf() const { return std::holds_alternative<T>(value); }

  bool is_vector() const {
    return std::holds_alternative<std::vector<Nest<T>>>(value);
  }

  bool is_map() const {
    return std::holds_alternative<std::map<std::string, Nest<T>>>(value);
  }

  std::vector<Nest<T>> &get_vector() {
    return std::get<std::vector<Nest<T>>>(value);
  }

  const std::vector<Nest<T>> &get_vector() const {
    return std::get<std::vector<Nest<T>>>(value);
  }

  T &front() {
    return std::visit(overloaded{[](T &t) -> T & { return t; },
                                 [](std::vector<Nest> &v) -> T & {
                                   return v.front().front();
                                 },
                                 [](std::map<std::string, Nest> &m) -> T & {
                                   return m.begin()->second.front();
                                 }},
                      value);
  }

  const T &front() const {
    return std::visit(
        overloaded{[](const T &t) -> const T & { return t; },
                   [](const std::vector<Nest> &v) -> const T & {
                     return v.front().front();
                   },
                   [](const std::map<std::string, Nest> &m) -> const T & {
                     return m.cbegin()->second.front();
                   }},
        value);
  }

  bool empty() const {
    return std::visit(
        overloaded{[](const T &t) { return false; },
                   [](const std::vector<Nest> &v) {
                     return std::all_of(v.begin(), v.end(),
                                        [](auto &n) { return n.empty(); });
                   },
                   [](const std::map<std::string, Nest> &m) {
                     return std::all_of(m.begin(), m.end(), [](auto &p) {
                       return p.second.empty();
                     });
                   }},
        value);
  }

  template <typename Function>
  Nest<std::invoke_result_t<Function, T>> map(Function f) const {
    using S = std::invoke_result_t<Function, T>;
    return std::visit(overloaded{[&f](const T &t) { return Nest<S>(f(t)); },
                                 [&f](const std::vector<Nest> &v) {
                                   std::vector<Nest<S>> result;
                                   result.reserve(v.size());
                                   for (const Nest<T> &n : v) {
                                     result.emplace_back(n.map(f));
                                   }
                                   return Nest<S>(result);
                                 },
                                 [&f](const std::map<std::string, Nest> &m) {
                                   std::map<std::string, Nest<S>> result;
                                   for (const auto &p : m) {
                                     result.emplace_hint(result.end(), p.first,
                                                         p.second.map(f));
                                   }
                                   return Nest<S>(result);
                                 }},
                      value);
  }

  std::vector<T> flatten() const {
    std::vector<T> result;
    flatten(std::back_inserter(result));
    return result;
  }

  template <class OutputIt>
  OutputIt flatten(OutputIt first) const {
    std::visit(overloaded{
                   [&first](const T &t) { *first++ = t; },
                   [&first](const std::vector<Nest> &v) {
                     for (const Nest &n : v) {
                       n.flatten(first);
                     }
                   },
                   [&first](const std::map<std::string, Nest> &m) {
                     for (auto &p : m) {
                       p.second.flatten(first);
                     }
                   },
               },
               value);
    return first;
  }

  template <class InputIt>
  Nest pack_as(InputIt first, InputIt last) const {
    Nest result = pack_as(&first, last);
    if (first != last) {
      throw std::range_error("Nest didn't exhaust sequence");
    }
    return result;
  }

  template <class InputIt>
  Nest pack_as(InputIt *first, const InputIt &last) const {
    return std::visit(
        overloaded{[&first, &last](const T &) {
                     if (*first == last)
                       throw std::out_of_range("Too few elements in sequence");
                     return Nest(*(*first)++);
                   },
                   [&first, &last](const std::vector<Nest> &v) {
                     std::vector<Nest> result;
                     result.reserve(v.size());
                     for (const Nest &n : v) {
                       result.emplace_back(n.pack_as(first, last));
                     }
                     return Nest(result);
                   },
                   [&first, &last](const std::map<std::string, Nest> &m) {
                     std::map<std::string, Nest> result;
                     for (auto &p : m) {
                       result.emplace_hint(result.end(), p.first,
                                           p.second.pack_as(first, last));
                     }
                     return Nest(result);
                   }},
        value);
  }

  static Nest<std::vector<T>> zip(const std::vector<Nest<T>> &nests) {
    const int nests_size = nests.size();
    if (nests_size == 0) {
      throw std::invalid_argument("Expected at least one nest.");
    }
    Nest<std::vector<T>> expanded =
        nests.begin()->map([nests_size](const T &t) {
          std::vector<T> leaf;
          leaf.reserve(nests_size);
          return leaf;
        });
    for (const Nest<T> &n : nests) {
      for_each([](std::vector<T> &v, const T &t) { v.emplace_back(t); },
               expanded, n);
    }
    return expanded;
  }

  template <typename Function, typename T1, typename T2>
  static Nest<std::invoke_result_t<Function, T1, T2>> map2(
      Function f, const Nest<T1> &nest1, const Nest<T2> &nest2) {
    using S = std::invoke_result_t<Function, T1, T2>;
    return std::visit(
        overloaded{
            [&f](const T1 &t1, const T2 &t2) { return Nest<S>(f(t1, t2)); },
            [&f](const std::vector<Nest<T1>> &v1,
                 const std::vector<Nest<T2>> &v2) {
              auto size = v1.size();
              if (size != v2.size()) {
                throw std::invalid_argument(
                    "Expected vectors of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(v2.size()));
              }
              std::vector<Nest<S>> result;
              result.reserve(size);
              auto it1 = v1.begin();
              auto it2 = v2.begin();
              for (; it1 != v1.end(); ++it1, ++it2) {
                result.emplace_back(map2(f, *it1, *it2));
              }
              return Nest<S>(result);
            },
            [&f](const std::map<std::string, Nest<T1>> &m1,
                 const std::map<std::string, Nest<T2>> &m2) {
              auto size = m1.size();
              if (size != m2.size()) {
                throw std::invalid_argument(
                    "Expected maps of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(m2.size()));
              }
              std::map<std::string, Nest<S>> result;
              auto it1 = m1.begin();
              auto it2 = m2.begin();
              for (; it1 != m1.end(); ++it1, ++it2) {
                if ((*it1).first != (*it2).first) {
                  throw std::invalid_argument(
                      "Expected maps to have same keys but found '" +
                      (*it1).first + "' vs '" + (*it2).first + "'");
                }
                result.emplace_hint(result.end(), (*it1).first,
                                    map2(f, (*it1).second, (*it2).second));
              }
              return Nest<S>(result);
            },
            [](auto &&arg1, auto &&arg2) -> Nest<S> {
              throw std::invalid_argument("nests don't match");
            }},
        nest1.value, nest2.value);
  }

  template <class Function>
  Function for_each(Function f) {
    std::visit(overloaded{f,
                          [&f](std::vector<Nest> &v) {
                            for (Nest &n : v) {
                              n.for_each(f);
                            }
                          },
                          [&f](std::map<std::string, Nest> &m) {
                            for (auto &p : m) {
                              p.second.for_each(f);
                            }
                          }},
               value);

    return std::move(f);
  }

  template <typename Function, typename T1, typename T2>
  static void for_each(Function f, Nest<T1> &nest1, const Nest<T2> &nest2) {
    return std::visit(
        overloaded{
            [&f](T1 &t1, const T2 &t2) { f(t1, t2); },
            [&f](std::vector<Nest<T1>> &v1, const std::vector<Nest<T2>> &v2) {
              auto size = v1.size();
              if (size != v2.size()) {
                throw std::invalid_argument(
                    "Expected vectors of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(v2.size()));
              }
              auto it1 = v1.begin();
              auto it2 = v2.begin();
              for (; it1 != v1.end(); ++it1, ++it2) {
                for_each(f, *it1, *it2);
              }
            },
            [&f](std::map<std::string, Nest<T1>> &m1,
                 const std::map<std::string, Nest<T2>> &m2) {
              auto size = m1.size();
              if (size != m2.size()) {
                throw std::invalid_argument(
                    "Expected maps of same length but got " +
                    std::to_string(size) + " vs " + std::to_string(m2.size()));
              }
              auto it1 = m1.begin();
              auto it2 = m2.begin();
              for (; it1 != m1.end(); ++it1, ++it2) {
                if ((*it1).first != (*it2).first) {
                  throw std::invalid_argument(
                      "Expected maps to have same keys but found '" +
                      (*it1).first + "' vs '" + (*it2).first + "'");
                }
                for_each(f, (*it1).second, (*it2).second);
              }
            },
            [](auto &&arg1, auto &&arg2) {
              throw std::invalid_argument("nests don't match");
            }},
        nest1.value, nest2.value);
  }
};
}  // namespace nest
