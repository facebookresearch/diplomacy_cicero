/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <optional>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "adjacencies.h"
#include "checks.h"
#include "civil_disorder_distances.h"
#include "game_state.h"
#include "loc.h"
#include "power.h"
#include "util.h"

using namespace std;
using nlohmann::json;

namespace dipcc {

OwnedUnit GameState::get_unit_rooted(Loc root) const {
  OwnedUnit unit;
  for (Loc loc : expand_coasts(root)) {
    unit = this->get_unit(loc);
    if (unit.type != UnitType::NONE) {
      return unit;
    }
  }
  return unit; // None
}

void GameState::remove_unit_rooted(Loc root) {
  for (Loc loc : expand_coasts(root)) {
    units_.erase(loc);
  }
}

void GameState::set_unit(Power power, UnitType type, Loc loc) {
  units_[loc] = {power, type, loc};
  influence_[loc] = power;
}

void GameState::add_dislodged_unit(OwnedUnit unit, Loc dislodged_by) {
  JCHECK(unit.type != UnitType::NONE, "add_dislodged_unit NONE unit");
  dislodged_units_[unit] = dislodged_by;
}

void GameState::remove_dislodged_unit(OwnedUnit unit) {
  dislodged_units_.erase(unit);
}

vector<OwnedUnit> GameState::get_dislodged_units() const {
  vector<OwnedUnit> r;
  r.reserve(dislodged_units_.size());
  for (auto &it : dislodged_units_) {
    r.push_back(it.first);
  }
  return r;
}

void GameState::add_contested_loc(Loc loc) { contested_locs_.insert(loc); }

OwnedUnit GameState::get_unit(Loc loc) const {
  auto x = units_.find(loc);
  if (x == units_.end()) {
    return {Power::NONE, UnitType::NONE, loc};
  } else {
    return x->second;
  }
}

void GameState::set_center(Loc loc, Power power) {
  JCHECK(is_center(loc), "set_center " + loc_str(loc));
  centers_[loc] = power;
}

void GameState::recalculate_centers() {
  for (auto &p : units_) {
    Loc center = root_loc(p.first);
    if (is_center(center)) {
      centers_[center] = p.second.power;
    }
  }
}

bool GameState::any_sc_occupied_by_new_power() const {
  // If the year ended right now, would any SCs change hands?
  for (auto &p : units_) {
    Loc center = root_loc(p.first);
    if (is_center(center)) {
      auto it = centers_.find(center);
      if (it == centers_.end() || it->second != p.second.power) {
        return true;
      }
    }
  }
  return false;
}

Power GameState::get_winner() const {
  vector<int> n_centers(7, 0);
  for (auto &p : centers_) {
    n_centers[static_cast<int>(p.second) - 1] += 1;
  }
  for (int i = 0; i < 7; ++i) {
    if (n_centers[i] >= 18) {
      return POWERS[i];
    }
  }
  return Power::NONE;
}

const unordered_map<Power, vector<Loc>> &GameState::get_orderable_locations() {
  if (!orders_loaded_) {
    this->get_all_possible_orders();
    orders_loaded_ = true;
  }
  return orderable_locations_;
}

const unordered_map<Loc, set<Order>> &GameState::get_all_possible_orders() {
  if (!orders_loaded_) {
    if (phase_.phase_type == 'M') {
      load_all_possible_orders_m();
    } else if (phase_.phase_type == 'R') {
      load_all_possible_orders_r();
    } else if (phase_.phase_type == 'A') {
      load_all_possible_orders_a();
    }
    orders_loaded_ = true;
  }

  return all_possible_orders_;
}

void GameState::load_all_possible_orders_m() {
  JCHECK(phase_.phase_type == 'M', "load_all_possible_orders_m non-m phase");
  clear_all_possible_orders();

  vector<vector<Order>> move_orders_by_dest;
  std::set<Loc> global_fleets_visited;

  all_possible_orders_.reserve(LOCS.size() + 1);
  move_orders_by_dest.resize(LOCS.size() + 1);

  unordered_map<Power, set<Loc>> orderable_locations;

  // Determine all orders except support-moves
  for (const auto &it : units_) {
    Unit unit = it.second.unowned();
    JCHECK(unit.type != UnitType::NONE, "load_all_possible_orders_m NONE unit");
    orderable_locations[it.second.power].insert(unit.loc);
    auto &adj = unit.type == UnitType::ARMY ? ADJ_A : ADJ_F;
    auto &adj_coasts =
        unit.type == UnitType::ARMY ? ADJ_A_ALL_COASTS : ADJ_F_ALL_COASTS;
    set<Order> &unit_orders = all_possible_orders_[unit.loc];

    // Hold
    unit_orders.insert(Order(unit, OrderType::H));

    // Non-via moves
    for (auto adj_loc : adj[static_cast<size_t>(unit.loc)]) {
      move_orders_by_dest[static_cast<int>(adj_loc)].push_back(
          Order(unit, OrderType::M, adj_loc));
    }

    // Support-holds
    for (auto adj_loc : adj_coasts[static_cast<size_t>(unit.loc)]) {
      const Unit &adj_unit = get_unit(adj_loc).unowned();
      if (adj_unit.type != UnitType::NONE) {
        unit_orders.insert(Order(unit, OrderType::SH, adj_unit));

        // Accept e.g. "F BLA S F BUL" instead of "F BUL/SC"
        Loc adj_root = root_loc(adj_unit.loc);
        if (adj_root != adj_unit.loc) {
          unit_orders.insert(
              Order(unit, OrderType::SH, {adj_unit.type, adj_root}));
        }
      }
    }

    // Convoys + moves via
    if (unit.type == UnitType::FLEET && is_water(unit.loc)) {
      std::set<Unit> adj_armies;
      std::set<Unit> adj_fleets_todo{unit};
      std::set<Unit> local_fleets_visited;
      std::set<Loc> adj_coast_locs;
      while (adj_fleets_todo.size() > 0) {

        // Pop fleet to consider
        auto fleet_it = adj_fleets_todo.begin();
        Unit fleet = *fleet_it;
        adj_fleets_todo.erase(fleet_it);
        local_fleets_visited.insert(fleet);
        global_fleets_visited.insert(fleet.loc);

        for (Loc adj_loc : ADJ_F_ALL_COASTS[static_cast<size_t>(fleet.loc)]) {

          if (!is_water(adj_loc)) {
            // Possible destination loc
            adj_coast_locs.insert(root_loc(adj_loc));
          }

          Unit adj_unit = this->get_unit(adj_loc).unowned();
          if (adj_unit.type == UnitType::FLEET && is_water(adj_loc) &&
              global_fleets_visited.find(adj_loc) ==
                  global_fleets_visited.end()) {
            // Adjacent fleet that can chain convoy
            adj_fleets_todo.insert(adj_unit);
          } else if (adj_unit.type == UnitType::ARMY) {
            // Possible source army to convoy
            adj_armies.insert(adj_unit);
          }
        }
      }

      // Each adj_army can be convoyed to each adj_coast_loc via each
      // local_fleets_visited
      for (const Unit &army : adj_armies) {
        for (Loc dest : adj_coast_locs) {
          if (dest == army.loc) {
            continue;
          }
          move_orders_by_dest[static_cast<int>(dest)].push_back(
              Order(army, OrderType::M, dest, true));
          for (const Unit &convoy_fleet : local_fleets_visited) {
            all_possible_orders_[convoy_fleet.loc].insert(
                Order(convoy_fleet, OrderType::C, army, dest));
          }
        }
      }
    }
  }

  // Move move_orders to all_possible_orders_
  for (const auto &move_orders : move_orders_by_dest) {
    for (const Order &order : move_orders) {
      all_possible_orders_[order.get_unit().loc].insert(order);
    }
  }

  // Determine support moves
  for (const auto &it : units_) {
    auto &unit = it.second;
    auto &adj_coasts =
        unit.type == UnitType::ARMY ? ADJ_A_ALL_COASTS : ADJ_F_ALL_COASTS;
    set<Order> &unit_orders = all_possible_orders_[unit.loc];

    for (Loc dest : adj_coasts[static_cast<size_t>(unit.loc)]) {
      if (dest == unit.loc) {
        continue; // can't support self-dislodge
      }
      for (const Order &move_order :
           move_orders_by_dest[static_cast<int>(dest)]) {
        if (move_order.get_unit().loc == unit.loc) {
          continue; // can't support own move
        }

        unit_orders.insert(
            Order(unit, OrderType::SM, move_order.get_unit(), dest));

        // Accept e.g. "F BLA S F CON - BUL" instead of "BUL/SC"
        Loc dest_root = root_loc(dest);
        if (dest_root != dest) {
          unit_orders.insert(
              Order(unit, OrderType::SM, move_order.get_unit(), dest_root));
        }
      }
    }
  }

  copy_sorted_root_locs(orderable_locations, orderable_locations_);
}

void GameState::load_all_possible_orders_r() {
  JCHECK(this->phase_.phase_type == 'R', "load_all_possible_orders_r non-r");
  clear_all_possible_orders();

  unordered_map<Power, set<Loc>> orderable_locations;

  for (auto &p : dislodged_units_) {
    OwnedUnit unit = p.first;
    Loc dislodger_root = root_loc(p.second);
    set<Order> &retreats = all_possible_orders_[unit.loc];
    retreats.insert(Order(unit.unowned(), OrderType::D));
    orderable_locations[unit.power].insert(unit.loc);
    const auto &adj_locs =
        (unit.type == UnitType::ARMY ? ADJ_A
                                     : ADJ_F)[static_cast<int>(unit.loc)];
    for (Loc adj : adj_locs) {
      if (root_loc(adj) == dislodger_root) {
        // can't retreat to dislodger src
        continue;
      }
      if (set_contains(contested_locs_, root_loc(adj))) {
        // can't retreat to bounced loc
        continue;
      }
      if (this->get_unit_rooted(adj).type != UnitType::NONE) {
        // can't retreat to occupied loc
        continue;
      }

      retreats.insert(Order(unit.unowned(), OrderType::R, adj));
    }
  }

  copy_sorted_root_locs(orderable_locations, orderable_locations_);
}

void GameState::load_all_possible_orders_a() {
  clear_all_possible_orders();
  unordered_map<Power, set<Loc>> orderable_locations;

  vector<int> n_units(7, 0);
  vector<int> n_centers(7, 0);
  vector<bool> can_disband(7, false);
  n_builds_.resize(7);

  // Count units
  for (auto &p : units_) {
    n_units[static_cast<int>(p.second.power) - 1] += 1;
  }

  // Count centers
  for (auto &p : centers_) {
    n_centers[static_cast<int>(p.second) - 1] += 1;
  }

  // Determine who builds and who disbands
  for (int p = 0; p < 7; ++p) {
    Power power = POWERS[p];
    n_builds_[p] = n_centers[p] - n_units[p];
    DLOG(INFO) << "load_all_possible_orders_a " << power_str(power)
               << " centers=" << n_centers[p] << " units=" << n_units[p];

    if (n_builds_[p] > 0) {
      // add army builds
      for (Loc center : home_centers_army(power)) {
        if (centers_.at(center) == power &&
            get_unit_rooted(center).type == UnitType::NONE) {
          all_possible_orders_[center].insert(
              Order({UnitType::ARMY, center}, OrderType::B));
          orderable_locations[power].insert(root_loc(center));
        }
      }

      // add fleet builds
      for (Loc center : home_centers_fleet(power)) {
        if (centers_.at(root_loc(center)) == power &&
            get_unit_rooted(center).type == UnitType::NONE) {
          all_possible_orders_[center].insert(
              Order({UnitType::FLEET, center}, OrderType::B));
          orderable_locations[power].insert(root_loc(center));
        }
      }
    }

    if (n_builds_[p] < 0) {
      // mark to add disbands in a single loop through all units
      can_disband[p] = true;
    }
  }

  // add disbands
  for (auto &p : units_) {
    if (can_disband[static_cast<int>(p.second.power) - 1]) {
      all_possible_orders_[p.first].insert(
          Order(p.second.unowned(), OrderType::D));
      orderable_locations[p.second.power].insert(root_loc(p.first));
    }
  }

  // sort doesn't matter/apply in A-phase, but we use this function
  // just to do the set->vector copy
  copy_sorted_root_locs(orderable_locations, orderable_locations_);
}

void GameState::clear_all_possible_orders() {
  all_possible_orders_.clear();
  orderable_locations_.clear();
  orders_loaded_ = false;
}

// Set orderable_locations_ in LOCS order. Sort by coastal variant, although
// orderable_locations_ contains root locs! This is to avoid downstream bugs
// where we iterate through locs, produce an order for each, and then the
// orders are not LOCS-ordered (since orders use coastal variants).
void GameState::copy_sorted_root_locs(
    const unordered_map<Power, set<Loc>> &from,
    unordered_map<Power, vector<Loc>> &to) {
  for (auto &[power, locs_set] : from) {
    auto &output = to[power];
    output.reserve(locs_set.size());
    for (Loc loc : locs_set) {
      // sorted set -- iterating in LOCS order -- but push root_loc for mila
      // compat
      output.push_back(root_loc(loc));
    }
  }
}

GameState GameState::process(const unordered_map<Power, vector<Order>> &orders,
                             bool exception_on_convoy_paradox) {
  DLOG(INFO) << "Processing " << this->get_phase().to_string();
  DLOG(INFO) << "Orders:";
  for (auto &it : orders) {
    DLOG(INFO) << " " << power_str(it.first);
    for (auto &order : it.second) {
      DLOG(INFO) << "   " << order.to_string();
    }
  }

  if (!orders_loaded_) {
    this->get_all_possible_orders();
  }
  if (phase_.phase_type == 'M') {
    return process_m(orders, exception_on_convoy_paradox);
  } else if (phase_.phase_type == 'R') {
    return process_r(orders);
  } else if (phase_.phase_type == 'A') {
    return process_a(orders);
  } else {
    JFAIL("Cannot process phase: " + phase_.to_string());
  }
}

GameState
GameState::process_r(const unordered_map<Power, vector<Order>> &orders) {
  GameState next_state;
  next_state.set_phase(this->get_phase().next(false));
  next_state.set_units(this->get_units());
  next_state.set_centers(this->get_centers());

  map<OwnedUnit, Loc> dislodged_units(this->dislodged_units_);
  const auto &all_possible_orders(this->get_all_possible_orders());
  set<Loc> multiple_retreater_locs;

  for (const auto &p : orders) {
    Power power = p.first;
    for (const Order &order : p.second) {
      OwnedUnit unit = order.get_unit().owned_by(power);
      auto dislodged_it = dislodged_units.find(unit);
      if (dislodged_it == dislodged_units.end()) {
        LOG(WARNING) << "Unit not dislodged [" << power_str(power)
                     << "]: " << order.to_string();
        continue;
      }
      if (!safe_contains(all_possible_orders, order.get_unit().loc, order)) {
        LOG(WARNING) << "Invalid retreat order [" << power_str(power)
                     << "]: " << order.to_string();
        continue;
      }

      // retreat order is valid: mark so another valid order is not accepted
      dislodged_units.erase(dislodged_it);

      if (order.get_type() == OrderType::D ||
          set_contains(multiple_retreater_locs, root_loc(order.get_dest()))) {
        // do nothing: unit not added to next_state
      } else if (next_state.get_unit_rooted(order.get_dest()).type !=
                 UnitType::NONE) {
        // retreat order was allowed (so dest was previously unoccupied), but
        // dest is now occupied: another unit has already retreated there, so
        // disband both
        next_state.remove_unit_rooted(order.get_dest());
        multiple_retreater_locs.insert(root_loc(order.get_dest()));
      } else {
        next_state.set_unit(power, order.get_unit().type, order.get_dest());
      }
    }
  }

  if (next_state.get_phase().season == 'W') {
    next_state.maybe_skip_winter_or_finish();
  }
  return next_state;
}

GameState
GameState::process_a(const unordered_map<Power, vector<Order>> &orders) {
  GameState next_state;
  next_state.set_phase(this->get_phase().next(false));
  next_state.set_units(this->get_units());
  next_state.set_centers(this->get_centers());

  auto &all_possible_orders(this->get_all_possible_orders());
  vector<int> n_builds(n_builds_);

  for (auto &it : orders) {
    Power power = it.first;
    int p = static_cast<int>(power) - 1;

    for (const Order &order : it.second) {
      if ((order.get_type() == OrderType::B && n_builds[p] < 1) ||
          (order.get_type() == OrderType::D && n_builds[p] > -1)) {
        LOG(WARNING) << "Ignoring extra order: [" << power_str(power) << "] "
                     << order.to_string();
        continue;
      }

      if (!safe_contains(all_possible_orders, order.get_unit().loc, order)) {
        LOG(WARNING) << "Illegal order: " << order.to_string();
        continue;
      }

      if (order.get_type() == OrderType::B) {
        // handle build
        if (next_state.get_unit_rooted(order.get_unit().loc).type !=
            UnitType::NONE) {
          LOG(WARNING) << "Duplicate build: " << order.to_string();
          continue;
        }
        // do the build
        next_state.set_unit(power, order.get_unit().type, order.get_unit().loc);
        n_builds[p]--;
      } else {
        // handle disband
        Power owner = next_state.get_unit_rooted(order.get_unit().loc).power;
        if (owner != power) {
          // Also caused by multiple disbands of same unit
          // DLOG(INFO) << "Disband unowned unit: " << order.to_string() << " by
          // "
          //            << power_str(power) << ", unit owned by "
          //            << (owner == Power::NONE ? "NONE" : power_str(owner));
          continue;
        }
        // do the disband
        next_state.remove_unit_rooted(order.get_unit().loc);
        n_builds[p]++;
      }
    }
  }

  // Check for civil disorder
  for (int p = 0; p < 7; ++p) {
    if (n_builds[p] < 0) {
      next_state.do_civil_disorder(POWERS[p], -n_builds[p]);
    }
  }

  return next_state;
}

void GameState::do_civil_disorder(Power power, int n) {
  DLOG(INFO) << "Civil disorder: " << power_str(power) << " " << n;
  JCHECK(n > 0, "do_civil_disorder must remove > 0 units");

  map<int, set<Unit>> sorted_units; // distance -> units

  // sort units by distance to home centers
  for (auto &p : units_) {
    if (p.second.power == power) {
      Unit unit = p.second.unowned();
      auto &dist_map = unit.type == UnitType::ARMY
                           ? CIVIL_DISORDER_DISTS_ARMY.at(power)
                           : CIVIL_DISORDER_DISTS_FLEET.at(power);
      int dist = dist_map[static_cast<int>(unit.loc) - 1];
      sorted_units[dist].insert(unit);
    }
  }

  // Traverse in reverse order (higher distance first)
  for (auto it = sorted_units.rbegin(); it != sorted_units.rend(); ++it) {
    const set<Unit> &units = it->second;
    if (units.size() <= n) {
      // remove all units at this dist, no need to sort
      for (Unit unit : units) {
        JCHECK(units_.erase(unit.loc) == 1,
               "do_civil_disorder Not found: " + unit.to_string());
      }
      n -= units.size();
    } else {
      // remove top n units sorted as follows:
      // 1. remove fleets before armies
      // 2. remove in alpha order
      vector<Unit> fleets;
      vector<Unit> armies;
      for (Unit unit : units) {
        (unit.type == UnitType::ARMY ? armies : fleets).push_back(unit);
      }
      while (n > 0) {
        vector<Unit> &next_vec = fleets.size() > 0 ? fleets : armies;
        vector<Unit>::iterator best_it;
        int best_it_prio = 99999;
        for (auto it = next_vec.begin(); it != next_vec.end(); ++it) {
          int prio = LOC_ALPHA_IDX[it->loc];
          if (prio < best_it_prio) {
            best_it_prio = prio;
            best_it = it;
          }
        }
        JCHECK(units_.erase(best_it->loc) == 1,
               "do_civil_disorder Not found: " + best_it->to_string());
        n -= 1;
        next_vec.erase(best_it);
      }
    }
  }
}

void GameState::maybe_skip_winter_or_finish() {
  JCHECK(phase_.season == 'W', "maybe_skip_winter_or_finish non-w");
  recalculate_centers();
  Power winner = get_winner();
  if (winner != Power::NONE || phase_.year == MAX_YEAR) {
    DLOG(INFO) << "Game over! Winner: "
               << (winner != Power::NONE ? power_str(winner) : "NONE");
    phase_ = phase_.completed();
    return;
  }

  for (auto &p : get_all_possible_orders()) {
    if (p.second.size() > 0) {
      // at least one power has a build/disband, so don't skip winter
      return;
    }
  }

  // no power has a build/disband, so skip winter
  clear_all_possible_orders();
  set_phase(get_phase().next(false));
}

bool GameState::has_any_unoccupied_home(Power power) const {
  for (Loc loc : home_centers(power)) {
    if (this->get_unit_rooted(loc).type == UnitType::NONE) {
      return true;
    }
  }
  return false;
}

int GameState::get_n_builds(Power power) {
  if (phase_.season != 'W') {
    return 0;
  }
  if (!orders_loaded_) {
    get_all_possible_orders();
  }
  return n_builds_.at(static_cast<size_t>(power) - 1);
}

json GameState::to_json() {
  json j;
  auto &all_possible_orders(this->get_all_possible_orders());

  // builds
  for (Power power : POWERS) {
    int n_homes = 0;
    j["builds"][power_str(power)]["homes"] = vector<string>();
    if (this->get_phase().phase_type == 'A') {
      for (Loc center : home_centers(power)) {
        auto it = all_possible_orders.find(center);
        if (it != all_possible_orders.end() && it->second.size() > 0) {
          j["builds"][power_str(power)]["homes"].push_back(loc_str(center));
          n_homes++;
        }
      }
    }
    j["builds"][power_str(power)]["count"] =
        phase_.phase_type == 'A' ? std::min(this->get_n_builds(power), n_homes)
                                 : 0;
  }

  // centers
  for (const auto &p : centers_) {
    j["centers"][power_str(p.second)].push_back(loc_str(p.first));
  }

  // homes
  for (Power power : POWERS) {
    j["homes"][power_str(power)] = vector<string>();
    for (Loc center : home_centers(power)) {
      auto owner_it = this->get_centers().find(center);
      if (owner_it != this->get_centers().end() && owner_it->second == power) {
        j["homes"][power_str(power)].push_back(loc_str(center));
      }
    }
  }

  // name
  j["name"] = this->get_phase().to_string();

  // retreats
  for (Power power : POWERS) {
    j["retreats"][power_str(power)] =
        std::map<string, string>(); // init with arbitrary empty map
  }
  if (this->get_phase().phase_type == 'R') {
    for (OwnedUnit &unit : this->get_dislodged_units()) {
      auto key = unit.unowned().to_string();
      auto orders_it = all_possible_orders.find(unit.loc);
      JCHECK(orders_it != all_possible_orders.end(),
             "Dislodged unit has no retreat orders in to_json: " +
                 loc_str(unit.loc));
      j["retreats"][power_str(unit.power)][key] = vector<string>();
      for (auto &order : orders_it->second) {
        if (order.get_type() == OrderType::R) {
          j["retreats"][power_str(unit.power)][key].push_back(
              loc_str(order.get_dest()));
        }
      }
    }
  }

  // units
  for (Power power : POWERS) {
    j["units"][power_str(power)] = std::vector<string>();
  }
  for (const auto &p : units_) {
    j["units"][power_str(p.second.power)].push_back(
        p.second.unowned().to_string());
  }
  for (const auto &p : dislodged_units_) {
    j["units"][power_str(p.first.power)].push_back(
        "*" + p.first.unowned().to_string());
  }

  return j;
}

GameState::GameState(const json &j) {

  // name
  phase_ = Phase(j["name"]);

  // builds
  if (phase_.season == 'W') {
    n_builds_.clear();
  }

  // centers
  for (auto &it : j["centers"].items()) {
    Power power = power_from_str(it.key());
    for (const string &s : it.value()) {
      Loc center = loc_from_str(s);
      centers_[center] = power;
    }
  }

  // homes: nothing to do

  // retreats
  if (phase_.phase_type == 'R') {
    unordered_map<Power, set<Loc>> orderable_locations;
    for (Power power : POWERS) {
      auto power_s = power_str(power);
      if (j["retreats"].find(power_s) == j["retreats"].end()) {
        continue;
      }
      for (auto &it : j["retreats"][power_s].items()) {
        Unit unit = Unit(it.key());
        dislodged_units_[unit.owned_by(power)] = Loc::NONE;
        orderable_locations[power].insert(unit.loc);
        all_possible_orders_[unit.loc].insert(Order(unit, OrderType::D));
        for (const string &s : it.value()) {
          Loc dest = loc_from_str(s);
          all_possible_orders_[unit.loc].insert(
              Order(unit, OrderType::R, dest));
        }
      }
    }
    copy_sorted_root_locs(orderable_locations, orderable_locations_);
    orders_loaded_ = true;
  }

  // units
  for (auto &it : j["units"].items()) {
    Power power = power_from_str(it.key());
    for (const string &unit_s : it.value()) {
      if (unit_s.at(0) == '*') {
        continue;
      }
      Unit unit(unit_s);
      units_[unit.loc] = unit.owned_by(power);
      influence_[unit.loc] = power;
    }
  }
}

void GameState::debug_log_all_possible_orders() {
  LOG(INFO) << "ORDERABLE LOCATIONS";
  for (auto &it : this->get_orderable_locations()) {
    LOG(INFO) << "  " << power_str(it.first);
    for (Loc loc : it.second) {
      LOG(INFO) << "    " << loc_str(loc);
    }
  }
  LOG(INFO) << "ALL POSSIBLE ORDERS:";
  for (auto &it : this->get_all_possible_orders()) {
    LOG(INFO) << "  " << loc_str(it.first);
    for (const Order &order : it.second) {
      LOG(INFO) << "    " << order.to_string();
    }
  }
}

std::vector<float> GameState::get_scores(Scoring scoring_system) const {
  std::vector<float> scores(7, 0);

  // get SC counts
  for (auto &p : centers_) {
    scores[static_cast<size_t>(p.second) - 1] += 1;
  }

  // check for winner
  for (int i = 0; i < 7; ++i) {
    if (scores[i] > 17.5) {
      // there is a winner, return 1-hot
      for (int j = 0; j < 7; ++j) {
        scores[j] = i == j ? 1 : 0;
      }
      return scores;
    }
  }

  // no winner: score based on scoring system
  float total = 0;
  for (int i = 0; i < 7; ++i) {
    if (scoring_system == Scoring::SOS) {
      scores[i] = scores[i] * scores[i];
    } else if (scoring_system == Scoring::DSS) {
      scores[i] = scores[i] > 0 ? 1 : 0;
    } else {
      JFAIL("Unknown scoring method");
    }
    total += scores[i];
  }

  // normalize
  for (int i = 0; i < 7; ++i) {
    scores[i] /= total;
  }

  return scores;
}

namespace {
// Using a separate function for std::map to make code die if anyone will change
// underlying objects to std::unordered_set in the future.
template <typename K, typename V>
inline void hash_combine_map(std::size_t &seed, const std::map<K, V> &map) {
  for (const auto &[k, v] : map) {
    hash_combine(seed, k);
    hash_combine(seed, v);
  }
}
} // namespace

size_t GameState::compute_board_hash() const {
  size_t ret = 0;
  hash_combine(ret, phase_);
  hash_combine(ret, units_.size());
  hash_combine_map(ret, units_);
  hash_combine(ret, centers_.size());
  hash_combine_map(ret, centers_);
  if (phase_.phase_type == 'R') {
    hash_combine(ret, dislodged_units_.size());
    hash_combine_map(ret, dislodged_units_);
    hash_combine(ret, contested_locs_.size());
    for (const auto loc : contested_locs_)
      hash_combine(ret, loc);
  }
  return ret;
}

} // namespace dipcc
