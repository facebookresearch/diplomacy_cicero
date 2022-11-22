/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../cc/cfrstats.h"
#include "../cc/exceptions.h"
#include "../cc/game.h"
#include "../cc/loc.h"
#include "../cc/thread_pool.h"
#include "encoding.h"
#include "py_game_get_units.h"
#include "thread_pool.h"

namespace py = pybind11;
using namespace dipcc;

// #################################################
// IF YOU EDIT THIS FILES
//     DON'T FORGET TO UPDATE fairdiplomacy/pydipcc.pyi
// !
// #################################################

PYBIND11_MODULE(pydipcc, m) {
  // class Game
  py::class_<Game>(m, "Game")
      .def(py::init<int, bool>(), py::arg("draw_on_stalemate_years") = -1,
           py::arg("is_full_press") = true)
      .def(py::init<const Game &>())
      .def("process", &Game::process)
      .def("set_orders", &Game::set_orders)
      .def("set_all_orders", &Game::set_all_orders,
           "NOTE: Clears and replaces any existing staged orders for any power")
      .def("clear_orders", &Game::clear_orders)
      .def("get_state", &Game::py_get_state, py::return_value_policy::move)
      .def("get_all_possible_orders", &Game::py_get_all_possible_orders)
      // Returns a dict power -> list of orderable locations. The list will be
      // sorted in the same way as x_possible_actions.
      .def("get_orderable_locations", &Game::py_get_orderable_locations)
      .def("to_json", &Game::to_json)
      .def("from_json", &Game::from_json)
      .def("from_json_inplace",
           [](Game &this_game, const std::string &json_content) {
             this_game = Game::from_json(json_content);
           })
      .def("get_phase_history", &Game::get_phase_history,
           py::return_value_policy::move,
           "Gets the phase data for all past phases, not including the current "
           "staged phase.")
      .def("get_staged_phase_data", &Game::get_staged_phase_data,
           py::return_value_policy::move,
           "Gets the phase data for the current staged phase that is not "
           "processed yet.")

      .def("get_phase_data", &Game::get_phase_data,
           py::return_value_policy::move,
           "NOTE: get_phase_data, bizarrely, omits the staged orders and "
           "messages"
           "of the current phase. This can lead to unexpected bugs, for example"
           "attempting to walk through the phase-by-phase messages of a game "
           "by playing"
           "through get_phase_history and then get_phase_data will NOT find "
           "all messages."
           "Use get_all_phases or get_staged_phase_data, which do not have "
           "this behavior. ")
      .def("get_all_phases", &Game::get_all_phases,
           py::return_value_policy::move,
           "Gets the phase data for all past phases and the current staged "
           "phase.")
      .def("get_all_phase_names", &Game::get_all_phase_names,
           py::return_value_policy::move)
      .def_property_readonly("message_history", &Game::py_get_message_history,
                             py::return_value_policy::move)
      .def_property_readonly("messages", &Game::py_get_messages,
                             py::return_value_policy::move)
      .def("get_logs", &Game::py_get_logs, py::return_value_policy::move)
      .def("add_log", &Game::add_log)
      .def("add_message", &Game::py_add_message, py::arg("sender"),
           py::arg("recipient"), py::arg("body"),
           py::arg_v("time_sent",
                     "Time sent. Expects a Timestamp object or centiseconds."),
           py::arg_v("increment_on_collision", false,
                     "If the timestamp is already used, increment until an "
                     "unused timestamp is found"))
      .def("rolled_back_to_phase_start", &Game::rolled_back_to_phase_start)
      .def("rolled_back_to_phase_end", &Game::rolled_back_to_phase_end)
      .def("rolled_back_to_timestamp_start",
           &Game::rolled_back_to_timestamp_start)
      .def("rolled_back_to_timestamp_end", &Game::rolled_back_to_timestamp_end)
      .def("phase_of_last_message_at_or_before",
           &Game::py_phase_of_last_message_at_or_before)
      .def("rollback_messages_to_timestamp_start",
           &Game::rollback_messages_to_timestamp_start)
      .def("rollback_messages_to_timestamp_end",
           &Game::rollback_messages_to_timestamp_end)
      .def("delete_message_at_timestamp", &Game::delete_message_at_timestamp)
      .def("get_last_message_timestamp", &Game::get_last_message_timestamp)
      .def_property_readonly("is_game_done", &Game::is_game_done)
      .def_property_readonly("phase", &Game::get_phase_long)
      .def_property_readonly("current_short_phase", &Game::get_phase_short)
      .def_property_readonly("current_year", &Game::get_year)
      .def_readwrite("game_id", &Game::game_id)
      .def_property_readonly("is_full_press", &Game::is_full_press)
      .def_property_readonly("map_name", &Game::map_name)
      .def("get_current_phase", &Game::get_phase_short)       // mila compat
      .def_property_readonly("phase_type", &Game::phase_type) // mila compat
      .def("get_orders", &Game::py_get_orders)                // mila compat
      .def("get_units", &py_game_get_units,
           py::return_value_policy::move) // mila compat
      .def("get_scores", [](Game &game) { return game.get_scores(); })
      .def("get_scores",
           [](Game &game, int scoring_system) {
             return game.get_scores(static_cast<Scoring>(scoring_system));
           })
      .def("get_unit_power_at", &Game::get_unit_power_at)
      .def("get_unit_type_at", &Game::get_unit_type_at)
      .def("is_supply_center", &Game::is_supply_center)
      .def("get_supply_center_power", &Game::get_supply_center_power)
      .def_property_readonly_static(
          "SCORING_SOS",
          [](py::object /* self */) { return static_cast<int>(Scoring::SOS); })
      .def_property_readonly_static(
          "SCORING_DSS",
          [](py::object /* self */) { return static_cast<int>(Scoring::DSS); })
      .def("get_scoring_system",
           [](Game &game) {
             return static_cast<int>(game.get_scoring_system());
           })
      .def("set_scoring_system",
           [](Game &game, int scoring_system) {
             game.set_scoring_system(static_cast<Scoring>(scoring_system));
           })
      .def("clear_old_all_possible_orders",
           &Game::clear_old_all_possible_orders)
      .def("set_exception_on_convoy_paradox",
           &Game::set_exception_on_convoy_paradox)
      // Hash of the current state of the board, i.e., position of units.
      .def("compute_board_hash", &Game::compute_board_hash)
      // Hash of the whole history of valid orders.
      .def("compute_order_history_hash", &Game::compute_order_history_hash)
      .def("set_draw_on_stalemate_years", &Game::set_draw_on_stalemate_years)
      .def("get_consecutive_years_without_sc_change",
           &Game::get_consecutive_years_without_sc_change)
      .def("any_sc_occupied_by_new_power", &Game::any_sc_occupied_by_new_power)
      .def("get_alive_powers",
           [](Game &game) {
             const auto scores = game.get_scores(Scoring::DSS);
             std::vector<std::string> powers;
             for (size_t i = 0; i < scores.size(); ++i) {
               if (scores[i] >= 1e-3) {
                 powers.push_back(POWERS_STR[i]);
               }
             }
             return powers;
           })
      .def("get_alive_power_ids",
           [](Game &game) {
             const auto scores = game.get_scores(Scoring::DSS);
             std::vector<int> powers;
             for (size_t i = 0; i < scores.size(); ++i) {
               if (scores[i] >= 1e-3) {
                 powers.push_back(i);
               }
             }
             return powers;
           })
      .def("get_next_phase",
           [](Game &game, const std::string &p) {
             auto x = game.get_next_phase(Phase(p));
             return x ? static_cast<py::object>(py::str(x->to_string()))
                      : py::none();
           })
      .def("get_prev_phase",
           [](Game &game, const std::string &p) {
             auto x = game.get_prev_phase(Phase(p));
             return x ? static_cast<py::object>(py::str(x->to_string()))
                      : py::none();
           })
      .def("clone_n_times",
           [](Game &game, const int &n_repeats) {
             std::vector<Game> games;
             games.reserve(n_repeats);
             for (int i = 0; i < n_repeats; ++i) {
               games.emplace_back(game);
               games.back().game_id += "_" + std::to_string(i);
             }
             return games;
           })
      .def("set_metadata", &Game::set_metadata)
      .def("get_metadata", &Game::get_metadata)
      .def_property_readonly_static(
          "LOC_STRS", [](py::object /* self */) { return VALID_LOC_STRS; })
      .def("is_water",
           [](const std::string &loc_str) {
             Loc loc = loc_from_str(loc_str);
             return loc == Loc::NONE ? false : is_water(loc);
           })
      .def("is_coast",
           [](const std::string &loc_str) {
             Loc loc = loc_from_str(loc_str);
             return loc == Loc::NONE ? false : is_coast(loc);
           })
      .def("is_center", [](const std::string &loc_str) {
        Loc loc = loc_from_str(loc_str);
        return loc == Loc::NONE ? false : is_center(loc);
      });

  // class PhaseData
  py::class_<PhaseData>(m, "PhaseData")
      .def_property_readonly("name", &PhaseData::get_name)
      .def_property_readonly("state", &PhaseData::py_get_state)
      .def_property_readonly("orders", &PhaseData::py_get_orders)
      .def_property_readonly("messages", &PhaseData::py_get_messages)
      .def("get_scores",
           [](PhaseData &phase_data, int scoring_system) {
             return phase_data.get_scores(static_cast<Scoring>(scoring_system));
           })
      .def("to_dict", &PhaseData::to_dict);

  // class ThreadPool
  py::class_<ThreadPool, std::shared_ptr<ThreadPool>>(m, "ThreadPool")
      .def(py::init<size_t, std::unordered_map<std::string, int>, int>())
      .def("process_multi", &ThreadPool::process_multi)
      .def("encode_orders_single_strict", &py_thread_pool_encode_orders_strict)
      .def("encode_orders_single_tolerant",
           &py_thread_pool_encode_orders_tolerant)
      .def("encode_inputs_multi", &py_thread_pool_encode_inputs_multi)
      .def("encode_inputs_all_powers_multi",
           &py_thread_pool_encode_inputs_all_powers_multi)
      .def("encode_inputs_state_only_multi",
           &py_thread_pool_encode_inputs_state_only_multi)
      .def("decode_order_idxs", &py_decode_order_idxs)
      .def("decode_order_idxs_all_powers", &py_decode_order_idxs_all_powers);

  // encoding functions
  m.def("encode_board_state", &py_encode_board_state,
        py::return_value_policy::move);
  m.def("encode_board_state_from_json", &py_encode_board_state_from_json,
        py::return_value_policy::move);
  m.def("encode_board_state_from_phase", &py_encode_board_state_from_phase,
        py::return_value_policy::move);
  m.def("encode_board_state_pperm_matrices",
        &py_encode_board_state_pperm_matrices, py::return_value_policy::move);
  m.def("encoding_unit_ownership_idxs", &encoding_unit_ownership_idxs);
  m.def("encoding_sc_ownership_idxs", &encoding_sc_ownership_idxs);

  m.def("max_input_version", []() { return MAX_INPUT_VERSION; });
  m.def("board_state_enc_width", &board_state_enc_width);

  // class CFRStat
  py::class_<SinglePowerCFRStats>(m, "SinglePowerCFRStats")
      .def(py::init<bool, bool, bool, bool, double, double, double,
                    const std::vector<double> &>())
      .def("cur_iter_strategy", &SinglePowerCFRStats::cur_iter_strategy)
      .def("bp_strategy", &SinglePowerCFRStats::bp_strategy)
      .def("avg_strategy", &SinglePowerCFRStats::avg_strategy)
      .def("avg_action_utilities", &SinglePowerCFRStats::avg_action_utilities)
      .def("avg_action_utility", &SinglePowerCFRStats::avg_action_utility)
      .def("avg_action_regret", &SinglePowerCFRStats::avg_action_regret)
      .def("cur_iter_action_prob", &SinglePowerCFRStats::cur_iter_action_prob)
      .def("avg_action_prob", &SinglePowerCFRStats::avg_action_prob)
      .def("avg_utility", &SinglePowerCFRStats::avg_utility)
      .def("avg_utility_stdev", &SinglePowerCFRStats::avg_utility_stdev)
      .def("update", &SinglePowerCFRStats::update)
      .def_property_readonly_static(
          "ACCUMULATE_PREV_ITER",
          [](py::object /* self */) {
            return SinglePowerCFRStats::ACCUMULATE_PREV_ITER;
          })
      .def_property_readonly_static(
          "ACCUMULATE_BLUEPRINT",
          [](py::object /* self */) {
            return SinglePowerCFRStats::ACCUMULATE_BLUEPRINT;
          })
      //__getstate__ and __setstate__ mean that SinglePowerCFRStats are
      // pickleable.
      .def("__getstate__",
           [](const SinglePowerCFRStats &p) { return p.__getstate__(); })
      .def("__setstate__", [](SinglePowerCFRStats &p, py::handle state) {
        SinglePowerCFRStats::__setstate__(p, state);
      });

  // class CFRStats
  py::class_<CFRStats>(m, "CFRStats")
      .def(py::init<bool, bool, bool, bool, double,
                    const std::map<std::string, double>,
                    const std::map<std::string, double>,
                    const std::map<std::string, std::vector<double>> &>())
      .def("cur_iter_strategy", &CFRStats::cur_iter_strategy)
      .def("bp_strategy", &CFRStats::bp_strategy)
      .def("avg_strategy", &CFRStats::avg_strategy)
      .def("avg_action_utilities", &CFRStats::avg_action_utilities)
      .def("avg_action_utility", &CFRStats::avg_action_utility)
      .def("avg_action_regret", &CFRStats::avg_action_regret)
      .def("cur_iter_action_prob", &CFRStats::cur_iter_action_prob)
      .def("avg_action_prob", &CFRStats::avg_action_prob)
      .def("avg_utility", &CFRStats::avg_utility)
      .def("avg_utility_stdev", &CFRStats::avg_utility_stdev)
      .def("update", &CFRStats::update)
      .def_property_readonly_static(
          "ACCUMULATE_PREV_ITER",
          [](py::object /* self */) { return CFRStats::ACCUMULATE_PREV_ITER; })
      .def_property_readonly_static(
          "ACCUMULATE_BLUEPRINT",
          [](py::object /* self */) { return CFRStats::ACCUMULATE_BLUEPRINT; })
      //__getstate__ and __setstate__ mean that CFRStats are pickleable.
      .def("__getstate__", [](const CFRStats &p) { return p.__getstate__(); })
      .def("__setstate__", [](CFRStats &p, py::handle state) {
        CFRStats::__setstate__(p, state);
      });

  // Exceptions
  py::register_exception<ConvoyParadoxException>(m, "ConvoyParadoxException");
}
