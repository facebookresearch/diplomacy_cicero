/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "enums.h"
#include "game_state.h"
#include "hash.h"
#include "loc.h"
#include "message.h"
#include "order.h"
#include "phase.h"
#include "power.h"
#include "unit.h"

namespace dipcc {

class SinglePowerCFRStats {
  struct ActionData {
    // Probability of action under blueprint proposal policy.
    double bp_prob = 0.0;
    // Probability of action under the regret-matching or hedge strategy for
    // next iteration
    double next_prob = 0.0;
    // Sum of per-iteration strategy probabilities for this action, weighted by
    // cum_weight_
    double cum_prob = 0.0;
    // Sum of regrets for this action, weighted by cum_weight_
    double cum_regret = 0.0;
    // Sum of utilities for this action, weighted by cum_weight_
    double cum_utility = 0.0;

    ActionData();
    ActionData(double bp_prob);
  };

public:
  // Arguments:
  // use_linear_weighting - Weight iteration t by t, instead of uniformly.
  // use_optimistic_cfr - Only matters if not qre, the last iteration counts
  // double. qre - If true, use qre, else use cfr qre_target_blueprint - Only
  // matters if qre. Bias towards bp_prob instead of uniform. qre_eta - Only
  // matters if qre. Parameter that controls convergence of qre. qre_lambdas -
  // Only matters if qre: qre lambda which describes the strength of bias
  // towards uniform or blueprint. bp_action_relprobs - the vector of blueprint
  // probabilities of the plausible actions. All further functions in this class
  // that deal with vectors of per-action values will adhere to the same
  // ordering.
  SinglePowerCFRStats(bool use_linear_weighting, bool use_optimistic_cfr,
                      bool qre, bool qre_target_blueprint, double qre_eta,
                      double qre_lambda, double qre_entropy_factor,
                      const std::vector<double> &bp_action_relprobs);

  // Constructor for pybind pickling/unpickling (__getstate__ and __setstate__)
  SinglePowerCFRStats(bool use_linear_weighting, bool cfr_optimistic, bool qre,
                      bool qre_target_blueprint, double qre_eta);

  // Class-level constants intended to be passed in for update.
  static const int ACCUMULATE_PREV_ITER = 1001;
  static const int ACCUMULATE_BLUEPRINT = 1002;

  // Update stats for a given power after an iteration.
  // Arguments:
  // state_utility - the actual utility achieved on this iteration
  // action_utilities - the utility for each action for this player
  // which_strategy_to_accumulate - one of ACCUMULATE_PREV_ITER
  //  or ACCUMULATE_BLUEPRINT.
  // cfr_iter - the 0-indexed iteration of CFR just finished.
  void update(double state_utility, const std::vector<double> &action_utilities,
              int which_strategy_to_accumulate, int cfr_iter);

  // All of the below functions return probabilities/regrets/utilities for
  // actions in the same order as the order of bp_action_relprobs_by_power
  // passed in.
  std::vector<double> cur_iter_strategy() const;
  std::vector<double> bp_strategy(double temperature) const;
  std::vector<double> avg_strategy() const;

  std::vector<double> avg_action_utilities() const;
  double cur_iter_action_prob(int action_idx) const;
  double avg_action_prob(int action_idx) const;
  double avg_action_utility(int action_idx) const;
  double avg_action_regret(int action_idx) const;
  double avg_utility() const;
  double avg_utility_stdev() const;

  pybind11::object __getstate__() const;
  static void __setstate__(SinglePowerCFRStats &buf,
                           const pybind11::handle &state);

private:
  const bool use_linear_weighting_;
  const bool use_optimistic_cfr_;
  const bool qre_;
  const bool qre_target_blueprint_;
  const double qre_eta_;

  std::vector<ActionData> actions_;

  double cum_utility_;
  double cum_squtility_;
  double cum_weight_;
  double qre_lambda_;
  double qre_entropy_factor_;
};

class CFRStats {
public:
  // Arguments:
  // use_linear_weighting - Weight iteration t by t, instead of uniformly.
  // use_optimistic_cfr - Only matters if not qre, the last iteration counts
  // double. qre - If true, use qre, else use cfr qre_target_blueprint - Only
  // matters if qre. Bias towards bp_prob instead of uniform. qre_eta - Only
  // matters if qre. Parameter that controls convergence of qre.
  // power_qre_lambdas - Only matters if qre: Power to qre lambda which
  // describes the strength of bias towards uniform or blueprint.
  // bp_action_relprobs_by_power - For each power, the vector of
  // blueprint probabilities
  // of the plausible actions for that power. All further functions in this
  // class that deal with vectors of per-action values will adhere to the same
  // ordering.
  CFRStats(bool use_linear_weighting, bool cfr_optimistic, bool qre,
           bool qre_target_blueprint, double qre_eta,
           const std::map<std::string, double> &power_qre_lambdas,
           const std::map<std::string, double> &power_qre_entropy_factor,
           const std::map<std::string, std::vector<double>>
               &bp_action_relprobs_by_power);

  // Constructor for pybind pickling/unpickling (__getstate__ and __setstate__)
  CFRStats(std::unordered_map<std::string, SinglePowerCFRStats> &&power_stats);

  // Class-level constants intended to be passed in for update.
  static const int ACCUMULATE_PREV_ITER =
      SinglePowerCFRStats::ACCUMULATE_PREV_ITER;
  static const int ACCUMULATE_BLUEPRINT =
      SinglePowerCFRStats::ACCUMULATE_BLUEPRINT;

  // Update stats for a given power after an iteration.
  // Arguments:
  // state_utility - the actual utility achieved on this iteration
  // action_utilities - the utility for each action for this player
  // which_strategy_to_accumulate - one of ACCUMULATE_PREV_ITER
  //  or ACCUMULATE_BLUEPRINT.
  // cfr_iter - the 0-indexed iteration of CFR just finished.
  void update(const std::string &power, double state_utility,
              const std::vector<double> &action_utilities,
              int which_strategy_to_accumulate, int cfr_iter);

  // All of the below functions return probabilities/regrets/utilities for
  // actions in the same order as the order of bp_action_relprobs_by_power
  // passed in.
  std::vector<double> cur_iter_strategy(const std::string &power) const;
  std::vector<double> bp_strategy(const std::string &power,
                                  double temperature) const;
  std::vector<double> avg_strategy(const std::string &power) const;

  std::vector<double> avg_action_utilities(const std::string &power) const;
  double cur_iter_action_prob(const std::string &power, int action_idx) const;
  double avg_action_prob(const std::string &power, int action_idx) const;
  double avg_action_utility(const std::string &power, int action_idx) const;
  double avg_action_regret(const std::string &power, int action_idx) const;
  double avg_utility(const std::string &power) const;
  double avg_utility_stdev(const std::string &power) const;

  pybind11::object __getstate__() const;
  static void __setstate__(CFRStats &buf, const pybind11::handle &state);

private:
  std::unordered_map<std::string, SinglePowerCFRStats> power_stats_;
};

} // namespace dipcc
