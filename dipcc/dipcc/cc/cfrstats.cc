/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "cfrstats.h"

#include <cmath>

namespace dipcc {

SinglePowerCFRStats::ActionData::ActionData() {}

SinglePowerCFRStats::ActionData::ActionData(double b) : bp_prob(b) {}

SinglePowerCFRStats::SinglePowerCFRStats(bool use_linear_weighting,
                                         bool use_optimistic_cfr, bool qre,
                                         bool qre_target_blueprint,
                                         double qre_eta)
    : use_linear_weighting_(use_linear_weighting),
      use_optimistic_cfr_(use_optimistic_cfr), qre_(qre),
      qre_target_blueprint_(qre_target_blueprint), qre_eta_(qre_eta) {}

SinglePowerCFRStats::SinglePowerCFRStats(
    bool use_linear_weighting, bool use_optimistic_cfr, bool qre,
    bool qre_target_blueprint, double qre_eta, double qre_lambda,
    double qre_entropy_factor, const std::vector<double> &bp_action_relprobs)
    : use_linear_weighting_(use_linear_weighting),
      use_optimistic_cfr_(use_optimistic_cfr), qre_(qre),
      qre_target_blueprint_(qre_target_blueprint), qre_eta_(qre_eta),
      cum_utility_(0.0), cum_squtility_(0.0), cum_weight_(0.0),
      qre_lambda_(qre_lambda), qre_entropy_factor_(qre_entropy_factor) {
  actions_.clear();
  JCHECK(bp_action_relprobs.size() > 0,
         "Empty policy for power found, please add an empty action policy "
         "for that power");
  double sum_bp_relprob = 0.0;
  for (double bp_relprob : bp_action_relprobs) {
    JCHECK(std::isfinite(bp_relprob) && bp_relprob >= 0.0,
           "Blueprint probability is nan or <= 0.0");
    actions_.push_back(ActionData(bp_relprob));
    // Begin with uniform distribution if anyone asks for the
    // regret-matching-based strategy right away before any updates
    actions_[actions_.size() - 1].next_prob = 1.0 / bp_action_relprobs.size();
    sum_bp_relprob += bp_relprob;
  }
  // We tolerate blueprints that are unnormalized, and we simply normalize
  // them.
  JCHECK(sum_bp_relprob > 0.0,
         "Sum of blueprint policy probablities is <= 0.0");
  for (ActionData &action_data : actions_) {
    action_data.bp_prob /= sum_bp_relprob;
  }
}

void SinglePowerCFRStats::update(double state_utility,
                                 const std::vector<double> &action_utilities,
                                 int which_strategy_to_accumulate,
                                 int cfr_iter) {
  JCHECK(action_utilities.size() == actions_.size(),
         "passed in action_utilities has the wrong length");

  // Discount for linear cfr
  if (use_linear_weighting_) {
    double discount_factor = (cfr_iter + 0.000001) / (cfr_iter + 1.0);
    cum_utility_ *= discount_factor;
    cum_squtility_ *= discount_factor;
    cum_weight_ *= discount_factor;
    for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
      ActionData &action_data = actions_[action_idx];
      action_data.cum_prob *= discount_factor;
      action_data.cum_regret *= discount_factor;
      action_data.cum_utility *= discount_factor;
    }
  }

  // Accumulate all stats
  cum_utility_ += state_utility;
  cum_squtility_ += state_utility * state_utility;
  cum_weight_ += 1.0;
  for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
    ActionData &action_data = actions_[action_idx];
    action_data.cum_regret += action_utilities[action_idx] - state_utility;
    action_data.cum_utility += action_utilities[action_idx];
  }
  if (which_strategy_to_accumulate == ACCUMULATE_PREV_ITER) {
    for (ActionData &action_data : actions_)
      action_data.cum_prob += action_data.next_prob;
  } else if (which_strategy_to_accumulate == ACCUMULATE_BLUEPRINT) {
    for (ActionData &action_data : actions_)
      action_data.cum_prob += action_data.bp_prob;
  } else {
    JFAIL("which_strategy_to_accumulate must be one of ACCUMULATE_PREV_ITER or "
          "ACCUMULATE_BLUEPRINT");
  }

  // Recompute the next probabilites of actions, QRE
  if (qre_) {
    double avg_utility = cum_utility_ / cum_weight_;
    double avg_squtility = cum_squtility_ / cum_weight_;
    double stdev_utility =
        sqrt(std::max(0.0, avg_squtility - avg_utility * avg_utility));
    double t = cum_weight_;
    double eta = qre_eta_ / (3.0 * (stdev_utility + 1e-6) * sqrt(t));

    double max_logits = -1e100;
    for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
      ActionData &action_data = actions_[action_idx];

      // If qre_target_blueprint_ true, use blueprint.
      // If qre_target_blueprint_ false, use uniform over plausible actions.
      // This needs to be the log of a number that is *proportional* to the
      // probability of the action under the policy that qre is regularizing us
      // toward. So we don't need to bother normalizing.
      double target_log_prob =
          qre_target_blueprint_ ? log(action_data.bp_prob + 1e-50) : 0.0;

      double qre_lambda = qre_lambda_;
      double avg_utility = action_data.cum_utility / cum_weight_;
      double logits = (avg_utility + qre_lambda * (1.0 + target_log_prob)) /
                      (qre_lambda * qre_entropy_factor_ + 1.0 / eta / t);

      // Not actually a probability yet, just storing it here for now
      action_data.next_prob = logits;
      if (logits > max_logits)
        max_logits = logits;
    }

    // Now perform softmax
    double sum_relative_prob = 0.0;
    for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
      ActionData &action_data = actions_[action_idx];
      double logits = action_data.next_prob;
      double relative_prob = exp(logits - max_logits);
      action_data.next_prob = relative_prob;
      sum_relative_prob += relative_prob;
    }
    JCHECK(std::isfinite(sum_relative_prob) && sum_relative_prob > 0,
           "cfr produced nan or infinite probabilities");
    // And normalize
    for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
      ActionData &action_data = actions_[action_idx];
      action_data.next_prob /= sum_relative_prob;
    }
  }
  // Recompute the next probabilites of actions, SearchBot (regret matching)
  else {
    // Compute relative probabilities proportional to positive regret
    double sum_relative_prob = 0.0;
    for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
      ActionData &action_data = actions_[action_idx];
      double cum_regret = action_data.cum_regret;
      if (use_optimistic_cfr_)
        cum_regret += action_utilities[action_idx] - state_utility;
      double relative_prob = std::max(0.0, cum_regret);
      action_data.next_prob = relative_prob;
      sum_relative_prob += relative_prob;
    }

    JCHECK(std::isfinite(sum_relative_prob) && sum_relative_prob >= 0,
           "cfr produced nan or infinite probabilities");
    if (sum_relative_prob == 0.0) {
      // Handle case where there are no positive regrets
      int best_action_idx = -1;
      for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
        ActionData &action_data = actions_[action_idx];
        if (best_action_idx < 0 ||
            action_data.cum_regret > actions_[best_action_idx].cum_regret) {
          best_action_idx = action_idx;
        }
      }
      for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
        ActionData &action_data = actions_[action_idx];
        action_data.next_prob = (action_idx == best_action_idx ? 1.0 : 0.0);
      }
    } else {
      // Normalize normally
      for (int action_idx = 0; action_idx < actions_.size(); ++action_idx) {
        ActionData &action_data = actions_[action_idx];
        action_data.next_prob /= sum_relative_prob;
      }
    }
  }
}

std::vector<double> SinglePowerCFRStats::cur_iter_strategy() const {
  std::vector<double> ret;
  ret.reserve(actions_.size());
  for (const ActionData &action_data : actions_) {
    ret.push_back(action_data.next_prob);
  }
  return ret;
}

std::vector<double> SinglePowerCFRStats::bp_strategy(double temperature) const {
  std::vector<double> ret;
  ret.reserve(actions_.size());
  double max_bp_prob = 0.0;
  for (const ActionData &action_data : actions_) {
    if (action_data.bp_prob > max_bp_prob) {
      max_bp_prob = action_data.bp_prob;
    }
  }

  double sum_relprob = 0.0;
  for (const ActionData &action_data : actions_) {
    double relprob;
    if (temperature <= 0.0) {
      if (action_data.bp_prob == max_bp_prob)
        relprob = 1.0;
      else
        relprob = 0.0;
    } else {
      relprob = pow(action_data.bp_prob / max_bp_prob, 1.0 / temperature);
    }

    ret.push_back(relprob);
    sum_relprob += relprob;
  }
  if (sum_relprob <= 0.0) {
    for (int i = 0; i < ret.size(); ++i)
      ret[i] = 1.0 / ret.size();
  } else {
    for (int i = 0; i < ret.size(); ++i)
      ret[i] /= sum_relprob;
  }
  return ret;
}

std::vector<double> SinglePowerCFRStats::avg_strategy() const {
  std::vector<double> ret;
  ret.reserve(actions_.size());
  double sum_relprob = 0.0;
  for (const ActionData &action_data : actions_) {
    double relprob = action_data.cum_prob;
    ret.push_back(relprob);
    sum_relprob += relprob;
  }
  if (sum_relprob <= 0.0) {
    for (int i = 0; i < ret.size(); ++i)
      ret[i] = 1.0 / ret.size();
  } else {
    for (int i = 0; i < ret.size(); ++i)
      ret[i] /= sum_relprob;
  }
  return ret;
}

double SinglePowerCFRStats::avg_action_prob(int action_idx) const {
  JCHECK(action_idx >= 0 && action_idx < actions_.size(),
         "out of bounds action_idx");
  return actions_[action_idx].cum_prob / cum_weight_;
}

double SinglePowerCFRStats::cur_iter_action_prob(int action_idx) const {
  JCHECK(action_idx >= 0 && action_idx < actions_.size(),
         "out of bounds action_idx");
  return actions_[action_idx].next_prob;
}

std::vector<double> SinglePowerCFRStats::avg_action_utilities() const {
  std::vector<double> ret;
  ret.reserve(actions_.size());
  for (const ActionData &action_data : actions_) {
    ret.push_back(action_data.cum_utility / cum_weight_);
  }
  return ret;
}

double SinglePowerCFRStats::avg_action_utility(int action_idx) const {
  JCHECK(action_idx >= 0 && action_idx < actions_.size(),
         "out of bounds action_idx");
  const ActionData &action_data = actions_[action_idx];
  return action_data.cum_utility / cum_weight_;
}

double SinglePowerCFRStats::avg_action_regret(int action_idx) const {
  JCHECK(action_idx >= 0 && action_idx < actions_.size(),
         "out of bounds action_idx");
  const ActionData &action_data = actions_[action_idx];
  return action_data.cum_regret / cum_weight_;
}

double SinglePowerCFRStats::avg_utility() const {
  double avg_utility = cum_utility_ / cum_weight_;
  return avg_utility;
}

double SinglePowerCFRStats::avg_utility_stdev() const {
  double avg_utility = cum_utility_ / cum_weight_;
  double avg_squtility = cum_squtility_ / cum_weight_;
  double stdev_utility =
      sqrt(std::max(0.0, avg_squtility - avg_utility * avg_utility));
  return stdev_utility;
}

CFRStats::CFRStats(
    bool use_linear_weighting, bool use_optimistic_cfr, bool qre,
    bool qre_target_blueprint, double qre_eta,
    const std::map<std::string, double> &power_qre_lambdas,
    const std::map<std::string, double> &power_qre_entropy_factor,
    const std::map<std::string, std::vector<double>>
        &bp_action_relprobs_by_power) {
  for (int power_idx = 0; power_idx < NUM_POWERS; ++power_idx) {
    auto bp_action_relprobs_entry =
        bp_action_relprobs_by_power.find(POWERS_STR[power_idx]);
    if (bp_action_relprobs_entry != bp_action_relprobs_by_power.end()) {
      double qre_lambda = power_qre_lambdas.at(POWERS_STR[power_idx]);
      double qre_entropy_factor =
          power_qre_entropy_factor.at(POWERS_STR[power_idx]);
      const std::vector<double> &bp_action_relprobs =
          bp_action_relprobs_entry->second;
      power_stats_.insert(std::make_pair(
          POWERS_STR[power_idx],
          SinglePowerCFRStats(use_linear_weighting, use_optimistic_cfr, qre,
                              qre_target_blueprint, qre_eta, qre_lambda,
                              qre_entropy_factor, bp_action_relprobs)));
    }
  }
}

CFRStats::CFRStats(
    std::unordered_map<std::string, SinglePowerCFRStats> &&power_stats)
    : power_stats_(std::move(power_stats)) {}

std::vector<double>
CFRStats::cur_iter_strategy(const std::string &power) const {
  return power_stats_.at(power).cur_iter_strategy();
}

std::vector<double> CFRStats::bp_strategy(const std::string &power,
                                          double temperature) const {
  return power_stats_.at(power).bp_strategy(temperature);
}

std::vector<double> CFRStats::avg_strategy(const std::string &power) const {
  return power_stats_.at(power).avg_strategy();
}

double CFRStats::avg_action_prob(const std::string &power,
                                 int action_idx) const {
  return power_stats_.at(power).avg_action_prob(action_idx);
}

double CFRStats::cur_iter_action_prob(const std::string &power,
                                      int action_idx) const {
  return power_stats_.at(power).cur_iter_action_prob(action_idx);
}

std::vector<double>
CFRStats::avg_action_utilities(const std::string &power) const {
  return power_stats_.at(power).avg_action_utilities();
}

double CFRStats::avg_action_utility(const std::string &power,
                                    int action_idx) const {
  return power_stats_.at(power).avg_action_utility(action_idx);
}

double CFRStats::avg_action_regret(const std::string &power,
                                   int action_idx) const {
  return power_stats_.at(power).avg_action_regret(action_idx);
}

double CFRStats::avg_utility(const std::string &power) const {
  return power_stats_.at(power).avg_utility();
}

double CFRStats::avg_utility_stdev(const std::string &power) const {
  return power_stats_.at(power).avg_utility_stdev();
}

void CFRStats::update(const std::string &power, double state_utility,
                      const std::vector<double> &action_utilities,
                      int which_strategy_to_accumulate, int cfr_iter) {
  power_stats_.at(power).update(state_utility, action_utilities,
                                which_strategy_to_accumulate, cfr_iter);
}
} // namespace dipcc
