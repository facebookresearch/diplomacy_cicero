/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "../cc/cfrstats.h"

namespace dipcc {

namespace py = pybind11;

py::object SinglePowerCFRStats::__getstate__() const {
  py::dict state;
  state["single_power_cfrstats_version_"] = 2;
  state["use_linear_weighting_"] = use_linear_weighting_;
  state["use_optimistic_cfr_"] = use_optimistic_cfr_;
  state["qre_"] = qre_;
  state["qre_target_blueprint_"] = qre_target_blueprint_;
  state["qre_eta_"] = qre_eta_;

  state["cum_utility_"] = cum_utility_;
  state["cum_squtility_"] = cum_squtility_;
  state["cum_weight_"] = cum_weight_;
  state["qre_lambda_"] = qre_lambda_;
  state["qre_entropy_factor_"] = qre_entropy_factor_;

  py::list all_action_datas_state;
  for (const ActionData &action_data : actions_) {
    py::list action_data_state;
    action_data_state.append(action_data.bp_prob);
    action_data_state.append(action_data.next_prob);
    action_data_state.append(action_data.cum_prob);
    action_data_state.append(action_data.cum_regret);
    action_data_state.append(action_data.cum_utility);
    all_action_datas_state.append(action_data_state);
  }
  state["actions_"] = all_action_datas_state;
  return state;
}

void SinglePowerCFRStats::__setstate__(SinglePowerCFRStats &buf,
                                       const py::handle &state) {
  int64_t cfrstats_version_ =
      state["single_power_cfrstats_version_"].cast<int64_t>();
  JCHECK(cfrstats_version_ == 1 || cfrstats_version_ == 2,
         "attempting to unpickle incompatible singlepowercfrstats format");

  bool use_linear_weighting_ = state["use_linear_weighting_"].cast<bool>();
  bool use_optimistic_cfr_ = state["use_optimistic_cfr_"].cast<bool>();
  bool qre_ = state["qre_"].cast<bool>();
  bool qre_target_blueprint_ = state["qre_target_blueprint_"].cast<bool>();
  double qre_eta_ = state["qre_eta_"].cast<double>();

  // Construct object in-place inside buf as per
  // https://pybind11-jagerman.readthedocs.io/en/stable/advanced.html#pickling-support
  new (&buf) SinglePowerCFRStats(use_linear_weighting_, use_optimistic_cfr_,
                                 qre_, qre_target_blueprint_, qre_eta_);

  buf.cum_utility_ = state["cum_utility_"].cast<double>();
  buf.cum_squtility_ = state["cum_squtility_"].cast<double>();
  buf.cum_weight_ = state["cum_weight_"].cast<double>();
  buf.qre_lambda_ = state["qre_lambda_"].cast<double>();
  if (cfrstats_version_ >= 2)
    buf.qre_entropy_factor_ = state["qre_entropy_factor_"].cast<double>();
  else
    buf.qre_entropy_factor_ = 1.0;

  buf.actions_.clear();
  for (py::handle action_data_state : state["actions_"]) {
    ActionData action_data;
    action_data.bp_prob = action_data_state[py::int_(0)].cast<double>();
    action_data.next_prob = action_data_state[py::int_(1)].cast<double>();
    action_data.cum_prob = action_data_state[py::int_(2)].cast<double>();
    action_data.cum_regret = action_data_state[py::int_(3)].cast<double>();
    action_data.cum_utility = action_data_state[py::int_(4)].cast<double>();
    buf.actions_.push_back(action_data);
  }
}

py::object CFRStats::__getstate__() const {
  py::dict state;
  state["cfrstats_version_"] = 1;
  for (const std::pair<std::string, SinglePowerCFRStats> &key_stats :
       power_stats_) {
    state[key_stats.first.c_str()] = key_stats.second.__getstate__();
  }
  return state;
}

void CFRStats::__setstate__(CFRStats &buf, const py::handle &state) {
  int64_t cfrstats_version_ = state["cfrstats_version_"].cast<int64_t>();
  JCHECK(cfrstats_version_ == 1,
         "attempting to unpickle incompatible cfrstats format");

  std::unordered_map<std::string, SinglePowerCFRStats> power_stats;
  for (int power_idx = 0; power_idx < NUM_POWERS; ++power_idx) {
    if (state.contains(POWERS_STR[power_idx].c_str())) {
      // Allocate memory without calling constructor, since __setstate__ expects
      // a memory buffer for which the constructor hasn't been called.
      SinglePowerCFRStats *spbuf = reinterpret_cast<SinglePowerCFRStats *>(
          ::operator new(sizeof(SinglePowerCFRStats)));
      SinglePowerCFRStats::__setstate__(*spbuf,
                                        state[POWERS_STR[power_idx].c_str()]);
      power_stats.insert(std::make_pair(POWERS_STR[power_idx], *spbuf));
      delete spbuf;
    }
  }

  // Construct object in-place inside buf as per
  // https://pybind11-jagerman.readthedocs.io/en/stable/advanced.html#pickling-support
  new (&buf) CFRStats(std::move(power_stats));
}

}; // namespace dipcc
