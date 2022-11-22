/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <glog/logging.h>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "adjacencies.h"
#include "checks.h"
#include "exceptions.h"
#include "game_state.h"
#include "power.h"
#include "unit.h"
#include "util.h"

using namespace std;

namespace dipcc {

// Represents one unit's bid to occupy a loc
struct LocCandidate {
  Loc src;                // with coast
  Loc dest;               // with coast
  Power power;            // controlling power
  int min;                // min strength of bid
  int max;                // max strength of bid
  int min_pending_convoy; // added to min when convoy is confirmed
  int min_pending_h2h;    // added to min when proven to not be dislodged by h2h
  int min_pending_convoy_h2h; // pending both convoy and h2h result
  int dislodge_self_support; // support strength of a power to dislodge own unit
  bool via;                  // is via convoy?
  bool via_adj;              // is via convoy but non-convoy move also possible?
};
const LocCandidate NONE_LOC_CANDIDATE = {
    Loc::NONE, Loc::NONE, Power::NONE, 0, 0, 0, 0, 0, 0, false, false};

// Output of resolve()
struct Resolution {
  map<Loc, LocCandidate> winners;
  set<Loc> dislodged;
  set<Loc> contested;
};

struct MoveCycle {
  bool convoy_swap;
  set<Loc> locs;
};

struct UnresolvedSupport {
  Order order;
  Power supporter_power;
  bool pending_dislodge;
};

class LocCandidates {
public:
  void add_candidate(Loc dest, OwnedUnit unit, bool via, bool via_adj) {
    Loc dest_root = root_loc(dest);
    Loc src_root = root_loc(unit.loc);

    // check for head to head battle: unit @ dest is trying to move to src,
    // both not via convoy
    auto &inbound_cands = cands_[src_root];
    auto h2h_cand_it = inbound_cands.find(dest_root);
    bool is_h2h =
        h2h_cand_it != inbound_cands.end() && !via && !h2h_cand_it->second.via;

    unresolved_units_.insert(src_root);
    cands_[dest_root][src_root] = {unit.loc,
                                   dest,
                                   unit.power,
                                   int(!via && !is_h2h), // min
                                   1,                    // max
                                   int(via && !is_h2h),  // min_pending_convoy
                                   int(!via && is_h2h),  // min_pending_h2h
                                   int(via && is_h2h), // min_pending_convoy_h2h
                                   0,                  // dislodge_self_support
                                   via,
                                   via_adj};

    if (src_root != dest_root) {
      // move request
      move_reqs_[src_root] = root_loc(dest_root);
      cands_[src_root][src_root].min = 0;
    }

    if (is_h2h) {
      // other side is also now in a h2h battle, adjust min strengths
      // min_pending_convoy -> min_pending_convoy_h2h
      h2h_cand_it->second.min_pending_convoy_h2h +=
          h2h_cand_it->second.min_pending_convoy;
      h2h_cand_it->second.min_pending_convoy = 0;
      // min -> min_pending_h2h
      h2h_cand_it->second.min_pending_h2h += h2h_cand_it->second.min;
      h2h_cand_it->second.min = 0;

      // save h2h battle
      unresolved_h2h_[src_root] = dest_root;
      unresolved_h2h_[dest_root] = src_root;
    }
  }

  void add_support(Order &order, Power supporter_power) {
    Loc src_root = root_loc(order.get_target().loc);
    Loc dest_root = order.get_type() == OrderType::SM
                        ? root_loc(order.get_dest())
                        : root_loc(order.get_target().loc);
    DLOG(INFO) << "ADD SUPPORT " << src_root << " - " << dest_root;
    map<Loc, LocCandidate> &dest_cands = cands_.at(dest_root);
    LocCandidate &supportee = dest_cands.at(root_loc(src_root));

    if (supportee.min_pending_convoy > 0) {
      supportee.min_pending_convoy += 1;
    } else if (supportee.min_pending_h2h > 0) {
      supportee.min_pending_h2h += 1;
    } else if (supportee.min_pending_convoy_h2h > 0) {
      supportee.min_pending_convoy_h2h += 1;
    } else {
      supportee.min += 1;
    }
    supportee.max += 1;

    auto hold_unit_it = dest_cands.find(dest_root);
    if (hold_unit_it != dest_cands.end() && src_root != dest_root &&
        hold_unit_it->second.power == supporter_power) {
      // This is a support-move into a loc controlled by the supporter's power.
      // Mark this so we don't allow a same-power dislodge
      supportee.dislodge_self_support += 1;
    }
  }

  void add_unresolved_support(Order &order, Power supporter_power,
                              bool pending_dislodge) {
    if (order.get_type() == OrderType::SM) {
      // unresolved support-move
      Loc dest_root = root_loc(order.get_dest());
      Loc src_root = root_loc(order.get_target().loc);
      map<Loc, LocCandidate> &dest_cands = cands_.at(dest_root);
      LocCandidate &supportee = dest_cands.at(src_root);
      supportee.max += 1;
    } else {
      // unresolved support-hold
      Loc target_root = root_loc(order.get_target().loc);
      LocCandidate &supportee = cands_.at(target_root).at(target_root);
      supportee.max += 1;
    }

    unresolved_supports_[root_loc(order.get_unit().loc)] =
        UnresolvedSupport{order, supporter_power, pending_dislodge};
  }

  void _resolve_support_if_exists(Resolution &r, Loc supporter_loc) {
    auto it = unresolved_supports_.find(supporter_loc);
    if (it != unresolved_supports_.end()) {
      UnresolvedSupport unresolved_support = it->second;
      remove_unresolved_support(unresolved_support.order);
      Loc dest = unresolved_support.order.get_type() == OrderType::SM
                     ? unresolved_support.order.get_dest()
                     : root_loc(unresolved_support.order.get_target().loc);
      if (!set_contains(r.contested, dest) && !map_contains(r.winners, dest)) {
        DLOG(INFO) << "CONFIRM SUPPORT: " << supporter_loc;
        add_support(unresolved_support.order,
                    unresolved_support.supporter_power);
      } else {
        // Skip adding support since the support dest is already resolved
        DLOG(INFO) << "SKIP CONFIRM SUPPORT FOR RESOLVED DEST: " << dest;
      }
    }
  }

  // Remove any unresolved support, unless it supports an attack on an attacker
  // from from_loc, e.g. if "A S B - C" and "C - A", then this support is in
  // defence and is not broken
  bool remove_unresolved_support_except_defence_from(Loc supporter_loc,
                                                     Loc from_loc) {
    auto it = unresolved_supports_.find(root_loc(supporter_loc));
    if (it == unresolved_supports_.end()) {
      // no support: nothing to do
      return false;
    }
    auto &order = it->second.order;
    if (order.get_type() == OrderType::SH) {
      // support-hold is broken
      remove_unresolved_support(order);
      return true;
    }
    if (root_loc(order.get_dest()) != from_loc) {
      // support-move dest is not from_loc
      remove_unresolved_support(order);
      return true;
    }
    auto support_dest_dest_it = move_reqs_.find(root_loc(order.get_dest()));
    if (support_dest_dest_it == move_reqs_.end() ||
        root_loc(support_dest_dest_it->second) != supporter_loc) {
      // support-move against unit not attacking supporter: broken
      remove_unresolved_support(order);
      return true;
    }
    // support is in defence! do not remove it
    return false;
  }

  bool remove_unresolved_support(Loc supporter_loc) {
    auto it = unresolved_supports_.find(root_loc(supporter_loc));
    if (it != unresolved_supports_.end()) {
      remove_unresolved_support(it->second.order);
      return true;
    }
    return false;
  }

  void remove_unresolved_support(Order &support_order) {
    if (support_order.get_type() == OrderType::SM) {
      // support-move
      Loc dest_root = root_loc(support_order.get_dest());
      Loc src_root = root_loc(support_order.get_target().loc);
      map<Loc, LocCandidate> &dest_cands = cands_.at(dest_root);
      LocCandidate &supportee = dest_cands.at(src_root);
      supportee.max -= 1;
    } else {
      // support-hold
      Loc target_root = root_loc(support_order.get_target().loc);
      LocCandidate &supportee = cands_.at(target_root).at(target_root);
      supportee.max -= 1;
    }

    unresolved_supports_.erase(root_loc(support_order.get_unit().loc));
  }

  void add_convoy_order(const Order &order) {
    maybe_convoy_orders_by_fleet_[order.get_unit().loc] = order;
    maybe_convoy_orders_by_dest_[root_loc(order.get_dest())].insert(order);
  }

  vector<LocCandidate> get_move_candidates(Loc dest) {
    dest = root_loc(dest);
    vector<LocCandidate> r;
    auto it = cands_.find(dest);
    if (it == cands_.end()) {
      return r;
    }
    for (auto &jt : it->second) {
      Loc src = jt.first;
      if (src != dest) {
        // include only move candidates, not hold
        r.push_back(jt.second);
      }
    }
    return r;
  }

  void log() {
    DLOG(INFO) << "Adjudicator State:";
    for (auto &it : cands_) {
      if (loc_prev_str_[root_loc(it.first)] > 0) {
        DLOG(INFO) << " Dest: " << loc_str(it.first)
                   << ", prev=" << loc_prev_str_[root_loc(it.first)];
      } else {
        DLOG(INFO) << " Dest: " << loc_str(it.first);
      }

      for (auto &jt : it.second) {
        DLOG(INFO) << "   " << loc_str(jt.first) << " " << jt.second.min << " "
                   << jt.second.max << " / " << jt.second.min_pending_convoy
                   << " " << jt.second.min_pending_h2h << " "
                   << jt.second.min_pending_convoy_h2h << " / "
                   << jt.second.dislodge_self_support;
      }
    }
    DLOG(INFO) << "Unresolved supports:";
    for (auto &it : unresolved_supports_) {
      DLOG(INFO) << " " << it.first << ": " << it.second.order << " "
                 << power_str(it.second.supporter_power) << " pending dislodge "
                 << it.second.pending_dislodge;
    }
    DLOG(INFO) << "Unconfirmed convoy fleets:";
    for (auto &it : maybe_convoy_orders_by_fleet_) {
      DLOG(INFO) << " " << it.first << ": " << it.second.to_string();
    }
    DLOG(INFO) << "Confirmed convoy fleets:";
    for (auto &it : confirmed_convoy_fleets_) {
      for (Loc fleet_loc : it.second) {
        DLOG(INFO) << " " << fleet_loc << ": " << it.first;
      }
    }
  }

  Resolution resolve() {
    Resolution r;

    for (int i = 1; i < 100; i++) {
      bool change_this_iter = false;

      for (auto &it : cands_) {
        Loc dest = it.first;
        map<Loc, LocCandidate> &loc_cands = it.second;

        change_this_iter |= _try_resolve_loc(r, dest, loc_cands);
      }

      if (!change_this_iter &&
          (unresolved_self_dislodges_.size() > 0 ||
           unresolved_self_support_dislodges_.size() > 0)) {
        DLOG(INFO) << "process() converged after " << i << " iterations with "
                   << unresolved_self_dislodges_.size()
                   << " unresolved self-dislodges and "
                   << unresolved_self_support_dislodges_.size()
                   << " unresolved self-support dislodges";
        this->log();
        _clear_unresolved_self_dislodges(r);
        continue; // keep iterating with cleared self dislodges
      }

      if (!change_this_iter && maybe_convoy_orders_by_fleet_.size() > 0) {
        DLOG(INFO) << "process() converged after " << i << " iterations with "
                   << maybe_convoy_orders_by_fleet_.size()
                   << " unresolved convoys. Checking for paradox.";
        this->log();
        _clear_convoy_paradoxes(r);
        continue; // keep iterating with cleared paradoxes
      }

      if (!change_this_iter) {
        DLOG(INFO) << "process() converged after " << i << " iterations";
        this->log();
        _finalize_resolve(r);
        JCHECK(unresolved_units_.size() == 0,
               "Unresolved units after finalize resolve");
        return r;
      }
    }

    LOG(ERROR) << "Too many iterations, preparing to abort";
    this->log();
    throw "Aborting after too many iterations";
  }

  // One resolve iteration for one destination loc
  //
  // Returns true if a change was made
  bool _try_resolve_loc(Resolution &r, Loc dest,
                        map<Loc, LocCandidate> &loc_cands) {
    if (map_contains(r.winners, dest)) {
      // dest is already resolved
      return false;
    }

    if (loc_cands.size() == 1) {
      // Only one candidate for dest
      auto only_cand_it = loc_cands.begin();
      if (only_cand_it->second.min >= 1) {
        // only candidate has min str >= 1, candidate wins
        _resolve_winner(r, dest, only_cand_it->second);
        return true;
      }
    } else {
      // Multiple candidates for dest

      // If this loc is the site of a h2h, try to resolve it first
      auto h2h_it = unresolved_h2h_.find(dest);
      if (h2h_it != unresolved_h2h_.end()) {
        _try_resolve_h2h(dest, h2h_it->second);
      }

      // Loop through all candidate for dest, gathering data e.g.  the
      // largest min, largest max, and whether there is a unit engaging in
      // a h2h
      map<Loc, LocCandidate>::iterator largest_min_cand;
      int largest_min = -1;
      int largest_max = -1;
      for (auto loc_cand_it = loc_cands.begin(); loc_cand_it != loc_cands.end();
           loc_cand_it++) {
        int min = loc_cand_it->second.min;
        int max = loc_cand_it->second.max;
        if (min > largest_min) {
          largest_min = min;
          largest_min_cand = loc_cand_it;
        }
        if (max > largest_max) {
          largest_max = max;
        }
        if (loc_cand_it->second.src == dest) {
          // maybe update holder prev str
          _set_prev_str_to_max(dest, min);
        }
      }

      // if the largest max is 0, this loc is empty
      if (largest_max == 0) {
        r.winners[dest] = NONE_LOC_CANDIDATE;
        return true;
      }

      // check largest min/max against loc's prevent strength
      if (largest_max <= loc_prev_str_[dest]) {
        // no unit can beat the prev strength: either units holds, or we bounce
        if (cands_[dest][dest].min > 0) {
          _resolve_winner(r, dest, cands_[dest][dest]);
        } else {
          _resolve_bounce(r, dest);
        }
        return true;
      }
      if (largest_min < loc_prev_str_[dest]) {
        // no unit's min beats prev strength, but >=1 units' max beats prev
        // strength: we are undecided
        return false;
      }

      // if largest min dominates other units' max, this unit wins
      bool is_winner = true;
      bool is_bounce = false;
      int other_units_max = -1;
      for (auto &loc_cand_it : loc_cands) {
        if (loc_cand_it.first == root_loc(largest_min_cand->second.src)) {
          continue;
        }
        int min = loc_cand_it.second.min;
        int max = loc_cand_it.second.max;
        if (max > other_units_max) {
          other_units_max = max;
        }
        if (max >= largest_min) {
          // largest min doesn't dominate this max, so no clear winner
          is_winner = false;
          if (max == min && max == largest_min && max == largest_max) {
            // two top candidates have exact same min/max strength: bounce
            is_bounce = true;
            break;
          }
        }
      }

      if (is_winner) {
        if (largest_min_cand->second.min -
                largest_min_cand->second.dislodge_self_support <=
            other_units_max) {
          // Winning candidate does not win without self-power support. If
          // this would cause a dislodge, remove support and recalculate to
          // avoid self-dislodge. If the unit currently at dest
          // successfully moves elsewhere, this is not a self-dislodge and
          // the support holds. See DATC 6.E.9
          return _handle_self_support_dislodge_support_winner(
              r, largest_min_cand->second);
        } else {
          auto hold_cand_it = loc_cands.find(dest);
          if (hold_cand_it != loc_cands.end() &&
              hold_cand_it->second.src != largest_min_cand->second.src &&
              hold_cand_it->second.power == largest_min_cand->second.power &&
              hold_cand_it->second.max > 0) {
            // unit of same power occupying dest; make sure we don't
            // self-dislodge
            if (hold_cand_it->second.min > 0) {
              // this is a self-dislodge: prevent it
              DLOG(INFO) << "Bounce to prevent self-dislodge: " << dest;
              _resolve_bounce(r, dest);
              return true;
            } else {
              // hold cand may still vacate: wait until that's resolved
              unresolved_self_dislodges_.insert(make_pair(
                  root_loc(largest_min_cand->second.src), root_loc(dest)));
              return false;
            }
          }

          _resolve_winner(r, dest, largest_min_cand->second);

          if (other_units_max == 0 &&
              maybe_convoy_orders_by_dest_[dest].size() == 0) {
            // no other unit managed to attack this loc: if the unit had an
            // unresolved support, resolve it
            _resolve_support_if_exists(r, dest);
          }

          return true;
        }
      } else if (is_bounce) {
        _resolve_bounce(r, dest);
        return true;
      }
    }

    // no change was made
    return false;
  }

  void _finalize_resolve(Resolution &r) {
    if (unresolved_units_.size() > 0) {
      DLOG(INFO) << unresolved_units_.size() << " unresolved units";
    }
    while (unresolved_units_.size() > 0) {
      Loc loc = *unresolved_units_.begin();
      DLOG(WARNING) << "Considering unresolved unit: " << loc_str(loc);
      if (move_reqs_.find(loc) == move_reqs_.end()) {
        // Unit tried to hold: must be dislodged
        JCHECK(map_contains(r.winners, loc),
               "Unit tried to hold but loc unresolved");
        JCHECK(root_loc(r.winners.at(loc).src) != loc,
               "Unit successfully held but still marked as unresolved");
        _resolve_dislodge(r, loc);
      } else {
        // Unit tried to move
        Loc dest = move_reqs_.at(loc);
        if (!map_contains(r.winners, dest)) {
          // Destination unresolved. Check for move cycle
          _check_and_resolve_move_cycle(r, loc);
        } else {
          // Destination resolved: must be dislodged
          JCHECK(root_loc(r.winners.at(dest).src) != loc,
                 "Unit successfully moved but still marked as unresolved");
          JCHECK(map_contains(r.winners, loc),
                 "Unit failed to move, but their loc remains unresolved. "
                 "Why didn't they hold?");
          JCHECK(root_loc(r.winners.at(loc).src) != loc,
                 "Unit failed to move, but successfully held. Why are "
                 "they unresolved?");
          _resolve_dislodge(r, loc);
        }
      }
    }
  }

  // Helper function for resolve(), called when a unit at `winner` will
  // move/hold to `dest`
  void _resolve_winner(Resolution &r, Loc dest, const LocCandidate &winner) {
    DLOG(INFO) << "RESOLVE WINNER " << loc_str(dest) << ": "
               << loc_str(winner.src);

    // Set output
    r.winners[dest] = winner;
    Loc winner_root = root_loc(winner.src);
    unresolved_units_.erase(winner_root);

    // If winner has an unresolved support, maybe resolve it
    _confirm_supporter_not_dislodged(r, winner_root);

    // If this was previously an unresolved self-support or self-dislodge
    // move, it has clearly been resolved
    unresolved_self_support_dislodges_.erase(make_pair(winner_root, dest));
    unresolved_self_dislodges_.erase(make_pair(winner_root, dest));

    // If winner was a convoying fleet, it is not dislodged, convoy fleet is
    // confirmed, check if army move is successful (if not already
    // successful)
    auto it = maybe_convoy_orders_by_fleet_.find(winner.src);
    if (it != maybe_convoy_orders_by_fleet_.end()) {
      Order &order = it->second;
      Loc army_src = root_loc(order.get_target().loc);
      Loc army_dest = root_loc(order.get_dest());
      maybe_convoy_orders_by_dest_[army_dest].erase(order);
      maybe_convoy_orders_by_fleet_.erase(it);
      LocCandidate &army_cand = cands_[army_dest][army_src];
      confirmed_convoy_fleets_[army_src].insert(winner.src);
      if (is_convoy_possible(army_src, army_dest, true)) {
        _confirm_convoy(army_cand);
      }
    }

    // Zero winner everywhere else
    // TODO: remove loop, check move_reqs_
    for (auto &it : cands_) {
      auto cand_elsewhere_it = it.second.find(winner_root);
      if (cand_elsewhere_it != it.second.end()) {
        // zero min and max elsewhere
        cand_elsewhere_it->second.min = 0;
        cand_elsewhere_it->second.max = 0;
      }
    }

    // Loser hold candidates are dislodged. Loser move candidates attempt to
    // hold their current position with strength 1.
    for (auto &loser : cands_[dest]) {
      Loc loser_loc = root_loc(loser.first);
      if (loser_loc == winner_root) {
        continue; // ignore winner
      }
      if (loser_loc == dest) {
        // loser formerly occupying dest must vacate or dislodge
        auto loser_dest_it = move_reqs_.find(loser_loc);
        if (loser_dest_it == move_reqs_.end()) {
          // loser was dislodged trying to hold
          _resolve_dislodge(r, loser_loc);
        } else if (loser_dest_it->second != winner_root) {
          // loser is trying to move somewhere other than the dislodger' src,
          // which means they cannot be dislodged by their h2h battle: resolve
          // their pending h2h strength
          _resolve_pending_h2h_strength(
              cands_.at(loser_dest_it->second).at(loser_loc));
        } else {
          // loser was trying to move to their dislodger's src.
          LocCandidate &loser_move_cand =
              cands_.at(loser_dest_it->second).at(loser_loc);
          if (!loser_move_cand.via && !winner.via) {
            // loser lost a h2h, and is dislodged
            _resolve_dislodge(r, loser_loc);
          }
        }
      } else if (loser.second.max > 0) {
        // loser mover now tries to hold their former position if it is not
        // already resolved, otherwise they are dislodged
        if (map_contains(r.winners, loser_loc)) {
          if (root_loc(r.winners.at(loser_loc).src) == loser_loc) {
            // unit has already successfully held: nothing to do here
          } else {
            // someone else has won the loser's loc: loser cannot hold, they
            // are
            // dislodged
            _resolve_dislodge(r, loser_loc);
          }
        } else if (set_contains(r.contested, loser_loc)) {
          // unit remains in bounced location
          _resolve_winner(r, loser_loc, cands_[loser_loc][loser_loc]);
        } else {
          // nobody has yet won loser's loc: they now contend to hold it
          cands_[loser_loc][loser_loc].min = 1;
          cands_[loser_loc][loser_loc].max = 1;
        }
      }
    }
  }

  // Helper function for resolve(), called when there is a bounce at `dest`.
  void _resolve_bounce(Resolution &r, Loc dest) {
    DLOG(INFO) << "RESOLVE BOUNCE " << loc_str(dest);

    // Make sure nobody retreats here
    r.contested.insert(dest);

    for (auto &cand : cands_[dest]) {
      Loc cand_loc = cand.first;

      if (cand_loc == dest) {
        // Unit already in this loc is not dislodged: if they have an
        // unresolved support, maybe confirm it.
        _confirm_supporter_not_dislodged(r, dest);

        // Unit is not dislodged: if they are in a H2H, their attack has
        // effect (see DATC 6.E.4)
        auto mov_it = move_reqs_.find(dest);
        if (mov_it != move_reqs_.end()) {
          if (map_contains(unresolved_h2h_, cand_loc)) {
            _resolve_pending_h2h_strength(cands_.at(mov_it->second).at(dest));
          } else {
            // h2h was already resolved, which means this cand lost a h2h. Add
            // pending strength to dest's prev strength
            _move_pending_h2h_str_to_prev(cands_.at(mov_it->second).at(dest));
          }
        }

        // Don't adjust strength for the unit already in this loc, who may
        // remain there.
        continue;
      }

      // All move candidates release their bid to hold dest
      // TODO: zero min_pending_h2h? can this scenario happen?
      cands_[dest][cand_loc].min = 0;
      cands_[dest][cand_loc].min_pending_convoy = 0;
      cands_[dest][cand_loc].max = 0;

      // Move candidates now attempt to hold their current position
      _failed_move_attempt_hold(cand_loc);
    }

    // Remove any unresolved self-dislodges at this location
    remove_self_dislodges_at_dest(unresolved_self_dislodges_, dest);
    remove_self_dislodges_at_dest(unresolved_self_support_dislodges_, dest);

    // Remove any unresolved self-dislodges at this location
    for (auto it = unresolved_self_dislodges_.begin();
         it != unresolved_self_dislodges_.end();) {
      if (root_loc(it->second) == root_loc(dest)) {
        it = unresolved_self_dislodges_.erase(it);
      } else {
        ++it;
      }
    }

    this->log();
  }

  void _resolve_dislodge(Resolution &r, Loc loc) {
    DLOG(INFO) << "DISLODGED " << loc_str(loc);
    r.dislodged.insert(loc);
    unresolved_units_.erase(loc);

    // Check if dislodged unit had an unresolved support, and if so resolve it
    // as a failure
    if (remove_unresolved_support(loc)) {
      DLOG(INFO) << "BROKEN SUPPORT: " << loc;
    }

    // Check if dislodged unit had any pending h2h strength vs. the dislodger,
    // and if so zero it (see DATC 6.E.1-5)
    auto move_it = move_reqs_.find(loc);
    if (move_it != move_reqs_.end()) {
      LocCandidate &dislodger_cand = r.winners.at(loc); // broke
      Loc dislodger_src = root_loc(dislodger_cand.src);
      LocCandidate &move_cand = cands_.at(move_it->second).at(loc);
      if (move_it->second == dislodger_src) {
        // dislodged unit has no effect on dislodger's loc
        _zero_pending_h2h_strength(move_cand);
      } else {
        // dislodged unit has effect on non-dislodger's loc
        _resolve_pending_h2h_strength(move_cand);
      }
    }

    // Check if dislodged unit was a convoying fleet, and if so check if the
    // convoy now fails
    _disable_convoy_fleet(r, loc);
  }

  MoveCycle _detect_unresolved_move_cycle(Loc loc) {
    MoveCycle cycle;
    cycle.convoy_swap = false;

    Loc dest = Loc::NONE;
    for (Loc cur = loc; dest != loc; cur = dest) {
      dest = move_reqs_.at(cur);
      if (!set_contains(unresolved_units_, dest) ||
          !map_contains(move_reqs_, dest)) {
        // not a move cycle, return empty
        cycle.locs.clear();
        return cycle;
      }
      cycle.convoy_swap |= cands_.at(dest).at(cur).via;

      if (dest != loc && set_contains(cycle.locs, cur)) {
        // found move cycle not containing loc, return empty
        cycle.locs.clear();
        return cycle;
      } else {
        cycle.locs.insert(cur);
      }
    }

    JCHECK(cycle.locs.size() >= 2, "Bad move cycle with size <2");
    return cycle;
  }

  void _check_and_resolve_move_cycle(Resolution &r, Loc loc) {
    MoveCycle cycle = _detect_unresolved_move_cycle(loc);
    if (cycle.locs.size() == 0) {
      // Not a move cycle, unit just failed to move and failed to
      // hold: dislodged
      _resolve_dislodge(r, loc);
      return;
    }

    // Swap if a h2h via swap or 3+ unit move cycle. Bounce for a 2-unit non-via
    // cycle (I'm 99% sure this should only happen for an adjacent convoy which
    // degrades to a non-via move, fwiw)
    bool swap = cycle.convoy_swap || cycle.locs.size() > 2;
    DLOG(INFO) << "RESOLVE " << cycle.locs.size()
               << " unit move cycle, via=" << cycle.convoy_swap;
    for (Loc d = loc;;) {
      Loc n = move_reqs_.at(d);
      if (swap) {
        r.winners[d] = cands_[n][d];
        DLOG(INFO) << "  " << d << " -> " << n;
      } else {
        r.winners[d] = cands_[d][d];
        DLOG(INFO) << "  " << d << " -> " << d;
      }
      unresolved_units_.erase(d);
      if (n == loc) {
        break;
      }
      d = n;
    }
  }

  // Shift min/max to indicate the winner of a h2h, but do not resolve the
  // loc! A unit can win a h2h but still lose the loc to a third unit.
  void _try_resolve_h2h(Loc loc_a, Loc loc_b) {
    auto cand_ab = cands_.at(loc_b).find(loc_a); // cand for move a -> b
    auto cand_ba = cands_.at(loc_a).find(loc_b); // cand for move b -> a

    int min_ab = cand_ab->second.min + cand_ab->second.min_pending_h2h;
    int min_ba = cand_ba->second.min + cand_ba->second.min_pending_h2h;

    int max_ab = cand_ab->second.max;
    int max_ba = cand_ba->second.max;

    if (min_ab > max_ba) {
      if (min_ab - cand_ab->second.dislodge_self_support > max_ba) {
        DLOG(INFO) << "H2H WINNER: " << cand_ab->second.src << " -> "
                   << cand_ba->second.src;
        _resolve_pending_h2h_strength(cand_ab->second);
        // zero max, but keep loser pending_h2h which may be
        // added to loc_prev_str_
        cand_ba->second.max = 0;
        _failed_move_attempt_hold(loc_b);
      } else {
        DLOG(INFO) << "H2H WINNER W/ SELF-DISLODGE: " << cand_ab->second.src
                   << " -> " << cand_ba->second.src;
        // a wins h2h only with self-dislodge support: h2h bounce
        _resolve_h2h_bounce(cand_ab->second, cand_ba->second);
      }

    } else if (min_ba > max_ab) {
      if (min_ba - cand_ba->second.dislodge_self_support > max_ab) {
        DLOG(INFO) << "H2H WINNER: " << cand_ba->second.src << " -> "
                   << cand_ab->second.src;
        _resolve_pending_h2h_strength(cand_ba->second);
        // zero max, but keep loser pending_h2h which may be
        // added to loc_prev_str_
        cand_ab->second.max = 0;
        _failed_move_attempt_hold(loc_a);
      } else {
        DLOG(INFO) << "H2H WINNER W/ SELF-DISLODGE: " << cand_ba->second.src
                   << " -> " << cand_ab->second.src;
        // b wins h2h only with self-dislodge support: h2h bounce
        _resolve_h2h_bounce(cand_ab->second, cand_ba->second);
      }

    } else if (min_ab == min_ba && max_ab == max_ba) {
      DLOG(INFO) << "H2H BOUNCE: " << loc_b << " <> " << loc_a;
      _resolve_h2h_bounce(cand_ab->second, cand_ba->second);
    } else {
      // unresolved, exit before cleanup
      return;
    }

    this->log();
    unresolved_h2h_.erase(loc_a);
    unresolved_h2h_.erase(loc_b);
  }

  void _set_prev_str_to_max(Loc loc, int val) {
    loc = root_loc(loc);
    int old_val = loc_prev_str_[loc];
    loc_prev_str_[loc] = val > old_val ? val : old_val;
  }

  void _move_pending_h2h_str_to_prev(LocCandidate &cand) {
    JCHECK(cand.min_pending_convoy_h2h == 0,
           "_move_pending_h2h_str_to_prev with pending_convoy_h2h");
    _set_prev_str_to_max(cand.dest, cand.min_pending_h2h);
    cand.min_pending_h2h = 0;
  }

  void _resolve_pending_h2h_strength(LocCandidate &cand) {
    cand.min += cand.min_pending_h2h;
    cand.min_pending_h2h = 0;
    cand.min_pending_convoy += cand.min_pending_convoy_h2h;
    cand.min_pending_convoy_h2h = 0;
  }

  void _zero_pending_h2h_strength(LocCandidate &cand) {
    // This unit will not win the h2h: reduce their max and pending min
    cand.max -= cand.min_pending_h2h + cand.min_pending_convoy_h2h;
    cand.min_pending_h2h = 0;
    cand.min_pending_convoy_h2h = 0;
  }

  void _resolve_h2h_bounce(LocCandidate &a, LocCandidate &b) {
    int a_min_pending_h2h = a.min + a.min_pending_h2h;
    int b_min_pending_h2h = b.min + b.min_pending_h2h;

    Loc a_dest = root_loc(a.dest);
    Loc b_dest = root_loc(b.dest);
    auto &a_dest_cands = cands_.at(a_dest);
    auto &b_dest_cands = cands_.at(b_dest);
    LocCandidate &a_dest_hold_cand = a_dest_cands.at(a_dest);
    LocCandidate &b_dest_hold_cand = b_dest_cands.at(b_dest);

    // move h2h strength to hold candidates instead: these units will not
    // swap,
    // but third party invaders must still beat these strengths to dislodge
    a_dest_hold_cand.min = a_min_pending_h2h;
    a_dest_hold_cand.max = a_min_pending_h2h;
    b_dest_hold_cand.min = b_min_pending_h2h;
    b_dest_hold_cand.max = b_min_pending_h2h;

    // zero/delete move cands
    auto &move_cand_a = a_dest_cands.at(b_dest);
    auto &move_cand_b = b_dest_cands.at(a_dest);
    move_cand_a.min = 0;
    move_cand_a.max = 0;
    move_cand_b.min = 0;
    move_cand_b.max = 0;
    move_reqs_.erase(a_dest);
    move_reqs_.erase(b_dest);
  }

  void _failed_move_attempt_hold(Loc loc) {
    // Failed move candidate now contends to hold their current position with
    // str=1. Ordinarily min will be 0 because this is a move candidate, but
    // we hack H2H bounces by pretending they're holding, in which case we
    // just want to keep the larger min/max that resulted from this hack.
    cands_[loc][loc].min = max(cands_[loc][loc].min, 1);
    cands_[loc][loc].max = max(cands_[loc][loc].max, 1);
  }

  void _disable_convoy_fleet(Resolution &r, Loc loc) {
    auto it = maybe_convoy_orders_by_fleet_.find(loc);
    if (it == maybe_convoy_orders_by_fleet_.end()) {
      return;
    }
    Order order = it->second;
    Loc src = root_loc(order.get_target().loc);
    Loc dest = root_loc(order.get_dest());
    maybe_convoy_orders_by_dest_[dest].erase(order);
    maybe_convoy_orders_by_fleet_.erase(it);
    if (!is_convoy_possible(order.get_target().loc, order.get_dest())) {
      bool via_adj = cands_[dest][src].via_adj;
      DLOG(INFO) << "BROKEN CONVOY " << src << " -> " << dest
                 << ", via_adj=" << via_adj;
      // erase all other pending convoys for this army
      erase_all_pending_convoys(src, dest);

      if (via_adj) {
        // convoy broken but army is adjacent to dest: just move normally
        _confirm_convoy(cands_[dest][src]);
        cands_[dest][src].via = false;
        cands_[dest][src].via_adj = false;
      } else {
        // army no longer attempts to move
        cands_[dest][src].min = 0;
        cands_[dest][src].max = 0;
        // army attempts to hold current loc
        cands_[src][src].min = 1;
        cands_[src][src].max = 1;
        // Even if unit at dest already won their loc, they may have had an
        // unresolved support which could now be confirmed.
        if (map_contains(r.winners, dest) &&
            root_loc(r.winners.at(dest).src) == dest &&
            _is_unresolved_supporter_and_all_remaining_convoys_do_not_cut(
                dest)) {
          _resolve_support_if_exists(r, dest);
        }
      }
    }
  }

  void erase_all_pending_convoys(Loc src, Loc dest) {
    set<Order> &maybe_dest_convoys = maybe_convoy_orders_by_dest_[dest];
    for (auto it = maybe_dest_convoys.begin();
         it != maybe_dest_convoys.end();) {
      if (it->get_target().loc == src) {
        Loc fleet_loc = it->get_unit().loc;
        DLOG(INFO) << "Disabling fleet for broken convoy: " << fleet_loc;
        it = maybe_dest_convoys.erase(it);
        JCHECK(maybe_convoy_orders_by_fleet_.erase(fleet_loc),
               "Disabled fleet missing in maybe_convoy_orders_by_fleet_");
      } else {
        ++it;
      }
    }
  }

  bool is_convoy_possible(Loc src, Loc dest, bool only_confirmed = false) {
    src = root_loc(src);

    // Compile fleets that are attempting to convoy this route
    const set<Loc> &confirmed_convoy_fleets = confirmed_convoy_fleets_[src];
    set<Loc> maybe_convoy_fleets;
    for (const Order &order : maybe_convoy_orders_by_dest_[dest]) {
      if (root_loc(order.get_target().loc) == src) {
        maybe_convoy_fleets.insert(order.get_unit().loc);
      }
    }

    set<Loc> todo;
    set<Loc> visited;

    // Initialize carefully: start with adjacent convoy fleets, not with src
    // directly, to ensure that VIA move goes through at least one fleet
    for (Loc _src : expand_coasts(src)) {
      for (Loc adj : ADJ_F[static_cast<size_t>(_src)]) {
        if (set_contains(confirmed_convoy_fleets, adj) ||
            (!only_confirmed && set_contains(maybe_convoy_fleets, adj))) {
          todo.insert(adj);
        }
      }
    }

    while (todo.size() > 0) {
      auto it = todo.begin();
      Loc loc = *it;
      todo.erase(it);
      visited.insert(loc);

      for (Loc current : ADJ_F_ALL_COASTS[static_cast<size_t>(loc)]) {
        if (root_loc(current) == root_loc(dest)) {
          return true;
        }
        if (set_contains(visited, current)) {
          continue;
        }
        if (set_contains(confirmed_convoy_fleets, current)) {
          // there is a fleet at current confirmed to convoy src -> dest
          todo.insert(current);
          continue;
        }
        if (only_confirmed) {
          continue;
        }

        // there is a fleet at current attempting to convoy src -> dest
        if (set_contains(maybe_convoy_fleets, current)) {
          todo.insert(current);
          continue;
        }
      }
    }
    return false;
  }

  // Unit @ loc is not dislodged: maybe confirm unresolved support
  void _confirm_supporter_not_dislodged(Resolution &r, Loc loc) {
    auto unresolved_support_it = unresolved_supports_.find(loc);
    if (unresolved_support_it == unresolved_supports_.end()) {
      return;
    }
    if (_is_unresolved_supporter_and_all_remaining_convoys_do_not_cut(loc)) {
      _resolve_support_if_exists(r, loc);
    } else {
      unresolved_support_it->second.pending_dislodge = false;
    }
  }

  // Return true if all convoy orders' src loc == loc
  bool _all_convoys_src_eq(const set<Order> &convoy_orders, Loc loc) {
    for (const Order &order : convoy_orders) {
      JCHECK(order.get_type() == OrderType::C,
             "_all_convoys_src_eq called with non-convoy");
      if (order.get_target().loc != loc) {
        return false;
      }
    }
    return true;
  }

  // Called to check when the only convoys that are unresolved at a location
  // should fail to cut support at that location. In particular, when all
  // convoys are of an army on which we are supporting an attack.
  bool _is_unresolved_supporter_and_all_remaining_convoys_do_not_cut(Loc loc) {
    auto unresolved_support_it = unresolved_supports_.find(loc);
    if (unresolved_support_it == unresolved_supports_.end()) {
      return false;
    }
    set<Order> &maybe_inbound_convoys = maybe_convoy_orders_by_dest_[loc];
    // Vacuously true if there are no convoys at all coming in
    if (maybe_inbound_convoys.size() == 0) {
      return true;
    }
    // Support may still be cut by convoyed army.
    // The only exception is if all inbound convoys are
    // ferrying an army on which we are supporting an attack. Check for this
    // case.
    Order &support_order = unresolved_support_it->second.order;
    if (support_order.get_type() == OrderType::SM &&
        _all_convoys_src_eq(maybe_inbound_convoys,
                            root_loc(support_order.get_dest()))) {
      return true;
    }

    return false;
  }

  // Called when a move candidate would win only with self-dislodge support.
  // Determine if the unit at loc would be dislodged, and if so prevent it.
  //
  // Return true if any change was made
  bool _handle_self_support_dislodge_support_winner(Resolution &r,
                                                    LocCandidate &cand) {
    Loc loc = root_loc(cand.dest);
    Loc src = root_loc(cand.src);
    DLOG(INFO) << "self-support dislodge candidate: " << src << " -> " << loc;
    auto dest_it = move_reqs_.find(loc);
    if (dest_it != move_reqs_.end()) {
      // unit @ loc was trying to move to dest
      Loc dest = root_loc(dest_it->second);
      auto dest_winner_it = r.winners.find(dest);
      if (dest_winner_it == r.winners.end() &&
          set_contains(r.contested, dest)) {
        // unit @ loc is trying to move to bounced loc, so we must prevent a
        // self-dislodge
        //
        // do nothing and pass through
      } else if (dest_winner_it == r.winners.end()) {
        // unit @ loc is trying to move to dest and may still win so this is
        // not necessarily a dislodge. Exit and wait for dest to be resolved.
        DLOG(INFO) << "self-support dislodge undecided";
        unresolved_self_support_dislodges_.insert(make_pair(src, loc));
        return false;
      } else if (root_loc(dest_winner_it->second.src) == root_loc(cand.dest)) {
        // unit at loc successfully vacated, so this is not a self-dislodge
        // and we can safely resolve the winning candidate
        DLOG(INFO) << "self-support dislodge resolve winner";
        _resolve_winner(r, loc, cand);
        unresolved_self_support_dislodges_.erase(make_pair(src, loc));
        return true;
      }
    }

    // Unit is trying to hold or lost a bid to move, so we must prevent a
    // self-dislodge
    DLOG(WARNING) << "Prevent self-support dislodge at " << loc;
    cand.min -= cand.dislodge_self_support;
    cand.max -= cand.dislodge_self_support;
    cand.dislodge_self_support = 0;
    unresolved_self_support_dislodges_.erase(make_pair(src, loc));
    _resolve_bounce(r, loc);
    return true;
  }

  // If we have made it to this point without resolution, remove self-dislodge
  // support entirely
  //
  void _clear_unresolved_self_dislodges(Resolution &r) {

    // Combine both unresolved self dislodge sets
    set<pair<Loc, Loc>> unresolved_self_dislodges(unresolved_self_dislodges_);
    for (auto &p : unresolved_self_support_dislodges_) {
      unresolved_self_dislodges.insert(p);
    }
    unresolved_self_dislodges_.clear();
    unresolved_self_support_dislodges_.clear();

    set<Loc> resolved_cycle_locs;
    vector<set<Loc>> maybe_valid_cycles;
    while (unresolved_self_dislodges.size() > 0) {
      auto it = unresolved_self_dislodges.begin();
      pair<Loc, Loc> p = *it;
      Loc src = p.first;
      Loc dest = p.second;
      DLOG(INFO) << "Consider resolving " << src << " -> " << dest;
      if (set_contains(resolved_cycle_locs, src)) {
        unresolved_self_dislodges.erase(it);
        continue;
      }
      MoveCycle cycle = _detect_unresolved_move_cycle(dest);
      if (set_contains(cycle.locs, src) &&
          (cycle.locs.size() >= 3 || cycle.convoy_swap)) {
        // self-dislodger is part of a move cycle, and therefore would not
        // dislodge dest if the cycle all moved. This is not a self-dislodge
        // unless another self-dislodger bounces this cycle. Save it as a
        // potentially valid cycle until all self-dislodge bounces have been
        // accounted for, and then allow the move cycle to continue
        DLOG(INFO) << "would-be self-dislodger part of move cycle: " << src
                   << " -> " << dest;
        maybe_valid_cycles.push_back(cycle.locs);
      } else {
        // self-dislodger would move into a move cycle, and therefore the
        // cycle will bounce
        DLOG(INFO) << "self-dislodger bounces move cycle: " << src << " -> "
                   << dest;
        _resolve_bounce(r, dest);
        for (Loc loc : cycle.locs) {
          resolved_cycle_locs.insert(loc);
        }
      }

      // erase by value, not by iterator, since set may have been modified in
      // the _resolve_* calls
      unresolved_self_dislodges.erase(p);
    }

    // Resolve valid move cycles if they were not later resolved by a bounce
    for (set<Loc> &cycle : maybe_valid_cycles) {
      Loc loc = *cycle.begin();
      if (!set_contains(resolved_cycle_locs, loc)) {
        Loc dest = move_reqs_.at(loc);
        _resolve_winner(r, dest, cands_.at(dest).at(loc));
        for (Loc c : cycle) {
          resolved_cycle_locs.insert(c);
        }
      }
    }
  }

  void remove_self_dislodges_at_dest(set<pair<Loc, Loc>> &unresolved_dislodges,
                                     Loc dest) {
    for (auto it = unresolved_dislodges.begin();
         it != unresolved_dislodges.end();) {
      if (root_loc(it->second) == root_loc(dest)) {
        it = unresolved_dislodges.erase(it);
      } else {
        ++it;
      }
    }
  }

  // This should be called after process iteration has converged with
  // unresolved convoys. This is indicative of a convoy paradox.  Resolve the
  // paradox according to the Szykman rule: paradoxical convoys have no
  // effect.
  //
  // See http://web.inter.nl.net/users/L.B.Kruijswijk/#4.A.2
  void _clear_convoy_paradoxes(Resolution &r) {
    while (maybe_convoy_orders_by_fleet_.size() > 0) {
      auto it = maybe_convoy_orders_by_fleet_.begin();
      Loc convoy_fleet = it->first;
      Order &convoy_order = it->second;
      DLOG(INFO) << "CONVOY PARADOX: " << convoy_order.to_string();
      if (exception_on_convoy_paradox_) {
        throw ConvoyParadoxException();
      }
      _disable_convoy_fleet(r, convoy_fleet);
    }
  }

  void _confirm_convoy(LocCandidate &army_cand) {
    Loc army_src = army_cand.src;
    Loc army_dest = army_cand.dest;
    DLOG(INFO) << "CONFIRM CONVOY " << army_src << " - " << army_dest;
    // pending convoy + h2h -> pending h2h
    army_cand.min_pending_h2h += army_cand.min_pending_convoy_h2h;
    army_cand.min_pending_convoy_h2h = 0;
    // pending convoy -> min
    army_cand.min += army_cand.min_pending_convoy;
    army_cand.min_pending_convoy = 0;
    // check for support cut
    if (remove_unresolved_support_except_defence_from(army_dest, army_src)) {
      DLOG(INFO) << "BROKEN SUPPORT: " << army_dest;
    }
  }

  /////////////////////
  // MEMBER VARIABLES
  /////////////////////

  // Loc candidates organized by (root) dest loc, then by (root) src loc
  // e.g. for the order "F STP/SC - BOT"
  // cands_[BOT][STP] = move candidate
  // cands_[STP][STP] = hold candidate (in case move fails)
  map<Loc, map<Loc, LocCandidate>> cands_;

  // Move candidates organized src -> dest
  map<Loc, Loc> move_reqs_;

  // Key is supporter loc
  map<Loc, UnresolvedSupport> unresolved_supports_;

  // Units who do not yet have a resolved destination
  set<Loc> unresolved_units_;

  // bidirectional map of unresolved h2h battles
  map<Loc, Loc> unresolved_h2h_;

  // unresolved self-support dislodge moves, src -> dest,
  // i.e. dislodges that succeed only with self-support
  set<pair<Loc, Loc>> unresolved_self_support_dislodges_;

  // unresolved self-dislodge moves, src -> dest,
  // i.e. dislodges of same-power unit
  set<pair<Loc, Loc>> unresolved_self_dislodges_;

  // Unresolved convoy orders organized by *fleet loc*
  map<Loc, Order> maybe_convoy_orders_by_fleet_;

  // Unresolved convoy orders organized by *dest loc*
  map<Loc, set<Order>> maybe_convoy_orders_by_dest_;

  // Non-dislodged fleets organized by *army loc*
  // Map of army loc -> set of fleet locs
  map<Loc, set<Loc>> confirmed_convoy_fleets_;

  // Minimum strength necessary to move to a loc.
  map<Loc, int> loc_prev_str_;

  // For debugging: if true, raise a custom exception when a convoy paradox is
  // encountered
  bool exception_on_convoy_paradox_ = false;
};

GameState GameState::process_m(
    const std::unordered_map<Power, std::vector<Order>> &orders,
    bool exception_on_convoy_paradox) {
  JCHECK(this->get_phase().phase_type == 'M',
         "Bad phase_type: " + this->get_phase().phase_type);
  DLOG(INFO) << "Process phase: " << this->get_phase().to_string();

  const unordered_map<Loc, set<Order>> &all_possible_orders(
      this->get_all_possible_orders());

  // Build up candidate data
  LocCandidates loc_candidates;
  vector<pair<Order, bool>> move_via_orders;
  vector<Order> support_orders;
  set<Loc> illegal_orderers;
  set<Loc> unconvoyed_movers;

  // Set debugging flags
  loc_candidates.exception_on_convoy_paradox_ = exception_on_convoy_paradox;

  // First add all existing units as candidates to remain in their
  // current location
  for (auto &it : this->units_) {
    Loc loc = it.first;
    OwnedUnit unit = it.second;
    loc_candidates.add_candidate(loc, unit, false, false);
  }

  // Organize orders by src loc
  unordered_map<Loc, Order> orders_by_src;
  for (auto &[power, porders] : orders) {
    for (const Order order : porders) {
      Loc loc = order.get_unit().loc;
      // check if correct power ordering unit
      if (this->get_unit(loc).power != power) {
        DLOG(WARNING) << power_str(power)
                      << " tried wrong-power order: " << order.to_string();
        continue;
      }
      orders_by_src[root_loc(loc)] = order;
    }
  }

  // Loop through all orders and build up data structures
  for (auto &[rloc, order] : orders_by_src) {
    // check if order is possible
    auto loc_possible_orders_it =
        all_possible_orders.find(order.get_unit().loc);
    if (loc_possible_orders_it == all_possible_orders.end() ||
        !set_contains(loc_possible_orders_it->second, order)) {
      if (is_implicit_via(order, all_possible_orders)) {
        // set via to explicitly true and move on
        DLOG(WARNING) << "Accepting implicit via for order: "
                      << order.to_string();
        order = order.with_via(true);
        orders_by_src[root_loc(order.get_unit().loc)] = order;
      } else {
        DLOG(WARNING) << "Order not possible: " << order.to_string();
        illegal_orderers.insert(root_loc(order.get_unit().loc));
        continue;
      }
    }

    // check if via move is to adjacent loc (i.e. non-via move also
    // allowed)
    bool via_adj =
        (order.get_via() &&
         loc_possible_orders_it != all_possible_orders.end() &&
         set_contains(loc_possible_orders_it->second, order.with_via(false)));

    // add all loc candidates and set aside supports
    if (order.get_type() == OrderType::H) {
      // do nothing, hold candidates already added
    } else if (order.get_type() == OrderType::M) {
      if (order.get_via()) {
        // Handle via moves after gathering convoy orders.
        move_via_orders.push_back(make_pair(order, via_adj));
      } else {
        // move to dest with max=1
        loc_candidates.add_candidate(order.get_dest(),
                                     this->get_unit(order.get_unit().loc),
                                     false, false);
      }
    } else if (order.get_type() == OrderType::SM ||
               order.get_type() == OrderType::SH) {
      // handle supports after determining which moves are legal
      support_orders.push_back(order);
    } else if (order.get_type() == OrderType::C) {
      auto target = orders_by_src.find(root_loc(order.get_target().loc));
      if (target != orders_by_src.end() &&
          target->second.get_type() == OrderType::M &&
          target->second.get_dest() == order.get_dest() &&
          (target->second.get_via() ||
           is_implicit_via(target->second, all_possible_orders))) {
        loc_candidates.add_convoy_order(order);
      } else {
        DLOG(WARNING) << "Uncoordinated convoy: " << order.to_string();
      }
    } else {
      throw("Can't yet categorize order: " + order.to_string());
    }
  }

  // Check for valid convoy path before adding move via order. Move may still
  // fail if a convoying fleet is dislodged
  for (auto &[order, via_adj] : move_via_orders) {
    if (loc_candidates.is_convoy_possible(root_loc(order.get_unit().loc),
                                          root_loc(order.get_dest()))) {
      loc_candidates.add_candidate(order.get_dest(),
                                   this->get_unit(order.get_unit().loc),
                                   order.get_via(), via_adj);
    } else {
      DLOG(INFO) << "Unconvoyed via move: " << order.to_string();
      loc_candidates.erase_all_pending_convoys(order.get_unit().loc,
                                               order.get_dest());

      if (via_adj) {
        DLOG(INFO) << "Unconvoyed via move converted to normal move: "
                   << order.to_string();
        loc_candidates.add_candidate(order.get_dest(),
                                     this->get_unit(order.get_unit().loc),
                                     false, false);
      } else {
        unconvoyed_movers.insert(order.get_unit().loc);
      }
    }
  }

  // Resolve supports
  for (Order &order : support_orders) {
    // Check for support coordination, e.g. that we are not support-holding a
    // unit that is moving, or support-moving a unit to the wrong destination
    auto target = orders_by_src.find(root_loc(order.get_target().loc));

    if (set_contains(unconvoyed_movers, root_loc(order.get_target().loc))) {
      DLOG(WARNING) << "Support of unconvoyed mover: " << order.to_string();
      continue;
    } else if (order.get_type() == OrderType::SM &&
               (target == orders_by_src.end() ||
                target->second.get_type() != OrderType::M ||
                // allow SM to specify exact dest or root dest
                (order.get_dest() != target->second.get_dest() &&
                 order.get_dest() != root_loc(target->second.get_dest())) ||
                set_contains(illegal_orderers,
                             root_loc(target->second.get_unit().loc)))) {
      DLOG(WARNING) << "Uncoordinated support-move: " << order.to_string();
      continue;
    } else if (order.get_type() == OrderType::SH &&
               (target != orders_by_src.end() &&
                target->second.get_type() == OrderType::M &&
                !set_contains(illegal_orderers,
                              root_loc(target->second.get_unit().loc)))) {
      DLOG(WARNING) << "Uncoordinated support-hold: " << order.to_string();
      continue;
    }

    // Check for support cuts.  Anyone (of a different power) trying to move
    // to any coastal variant is a cut candidate
    Power supporter_power = this->get_unit(order.get_unit().loc).power;
    set<Loc> cut_candidates;
    set<Loc> convoy_cut_candidates;
    for (Loc loc : expand_coasts(order.get_unit().loc)) {
      for (LocCandidate move_cand : loc_candidates.get_move_candidates(loc)) {
        if (move_cand.power == supporter_power) {
          // can't cut own support
          continue;
        }
        if (move_cand.via) {
          DLOG(INFO) << "Convoy cut candidate " << order.to_string() << " : "
                     << root_loc(move_cand.src);
          convoy_cut_candidates.insert(root_loc(move_cand.src));
        } else {
          DLOG(INFO) << "Cut candidate " << order.to_string() << " : "
                     << root_loc(move_cand.src);
          cut_candidates.insert(root_loc(move_cand.src));
        }
      }
    }

    if (order.get_type() == OrderType::SH) {
      //
      // handle support-hold
      //
      if (cut_candidates.size() == 0 && convoy_cut_candidates.size() == 0) {
        // support-hold is not cut, increase strength
        loc_candidates.add_support(order, supporter_power);
      } else if (cut_candidates.size() == 0 &&
                 convoy_cut_candidates.size() > 0) {
        // support cut depends on convoy
        loc_candidates.add_unresolved_support(order, supporter_power, false);
      } else {
        // cut_candidates > 0, support is cut
      }
    } else {
      //
      // handle support-move
      //
      if (cut_candidates.size() == 0 && convoy_cut_candidates.size() == 0) {
        // support-move is not cut, increase strength
        loc_candidates.add_support(order, supporter_power);
      } else if (cut_candidates.size() == 0 &&
                 convoy_cut_candidates.size() > 0) {
        // support cut depends on convoy
        loc_candidates.add_unresolved_support(order, supporter_power, false);
      } else if (cut_candidates.size() == 1) {
        // pending_dislodge: potential cutter is being attacked by this
        // support: support is conditional on dislodgedment (see DATC 6.D.17)
        if (root_loc(*cut_candidates.begin()) == root_loc(order.get_dest())) {
          loc_candidates.add_unresolved_support(order, supporter_power, true);
        }
      }
    }
  }

  // Resolve moves
  loc_candidates.log();
  auto resolved = loc_candidates.resolve();
  return build_next_state(resolved);
}

GameState GameState::build_next_state(const Resolution &r) const {

  GameState next;
  next.set_centers(this->get_centers());
  next.set_influence(this->get_influence());

  // Set units
  for (auto &it : r.winners) {
    LocCandidate cand = it.second;
    if (cand.src == Loc::NONE) {
      continue;
    }
    auto unit = this->get_unit(cand.src);
    JCHECK(unit.type != UnitType::NONE,
           "Bad: dest=" + loc_str(cand.dest) + " src=" + loc_str(cand.src));
    next.set_unit(unit.power, unit.type, cand.dest);
  }

  // Progress to R or M/A phase depending on dislodged units
  if (r.dislodged.size() > 0) {
    // Dislodged units: retreat phase
    next.set_phase(this->get_phase().next(true));

    for (Loc dislodged : r.dislodged) {
      auto &dislodger_cand = r.winners.at(dislodged);
      // hack: NONE dislodger loc if dislodged by convoy, since units may
      // retreat to dislodger src in this scenario
      Loc dislodger_loc = dislodger_cand.via ? Loc::NONE : dislodger_cand.src;
      next.add_dislodged_unit(this->get_unit_rooted(dislodged), dislodger_loc);
    }
    for (Loc contested : r.contested) {
      next.add_contested_loc(contested);
    }

    // Some units need to retreat, so progress to retreat phase
    return next;
  }

  // No dislodged units: move to next M/A phase
  next.set_phase(this->get_phase().next(false));
  if (next.get_phase().season == 'W') {
    next.maybe_skip_winter_or_finish();
  }

  return next;
}

} // namespace dipcc
