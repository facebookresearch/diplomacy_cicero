#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import unittest
import pickle
import math
import os

from fairdiplomacy.pydipcc import CFRStats, SinglePowerCFRStats
from fairdiplomacy.models.consts import POWERS


class TestCFRStatsPickle(unittest.TestCase):
    def test(self):
        use_linear_weighting = True
        cfr_optimistic = False
        qre = False
        qre_target_blueprint = True
        qre_eta = 2.0
        power_qre_lambdas = {power: 3.0 for power in POWERS}
        power_qre_entropy_factor = {power: 0.6 for power in POWERS}
        bp_action_relprobs_by_power = {power: [0.6, 0.4] for power in POWERS}
        stats = CFRStats(
            use_linear_weighting,
            cfr_optimistic,
            qre,
            qre_target_blueprint,
            qre_eta,
            power_qre_lambdas,
            power_qre_entropy_factor,
            bp_action_relprobs_by_power,
        )
        stats.update("FRANCE", 4.5, [1.0, 2.0], CFRStats.ACCUMULATE_PREV_ITER, 0)
        stats.update("FRANCE", 10.5, [10.0, 20.0], CFRStats.ACCUMULATE_PREV_ITER, 1)
        stats.update("ENGLAND", 15.6, [3.0, 7.0], CFRStats.ACCUMULATE_BLUEPRINT, 0)
        stats.update("RUSSIA", 15.6, [3.0, 7.0], CFRStats.ACCUMULATE_PREV_ITER, 0)
        dumped = pickle.dumps(stats)
        loaded = pickle.loads(dumped)

        assert loaded.bp_strategy("AUSTRIA", 1.0) == [0.6, 0.4]
        assert loaded.bp_strategy("FRANCE", 1.0) == [0.6, 0.4]
        assert math.isnan(loaded.avg_utility("GERMANY"))
        assert abs(loaded.avg_utility("FRANCE") - 8.5) <= 0.001
        assert loaded.avg_utility("ENGLAND") == 15.6
        assert abs(loaded.avg_action_utilities("FRANCE")[0] - 7.0) <= 0.001
        assert abs(loaded.avg_action_utilities("FRANCE")[1] - 14.0) <= 0.001
        assert loaded.avg_strategy("ENGLAND") == [0.6, 0.4]
        assert loaded.avg_strategy("RUSSIA") == [0.5, 0.5]

        assert loaded.__getstate__() == {
            "cfrstats_version_": 1,
            "TURKEY": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 0.0,
                "cum_squtility_": 0.0,
                "cum_weight_": 0.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.5, 0.0, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0, 0.0]],
            },
            "RUSSIA": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 15.6,
                "cum_squtility_": 243.35999999999999,
                "cum_weight_": 1.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.0, 0.5, -12.6, 3.0], [0.4, 1.0, 0.5, -8.6, 7.0]],
            },
            "ENGLAND": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 15.6,
                "cum_squtility_": 243.35999999999999,
                "cum_weight_": 1.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.0, 0.6, -12.6, 3.0], [0.4, 1.0, 0.4, -8.6, 7.0]],
            },
            "FRANCE": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 12.75000225,
                "cum_squtility_": 120.375010125,
                "cum_weight_": 1.5000005,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [
                    [0.6, 0.0, 0.25000025, -2.25000175, 10.5000005],
                    [0.4, 1.0, 1.25000025, 8.24999875, 21.000001],
                ],
            },
            "ITALY": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 0.0,
                "cum_squtility_": 0.0,
                "cum_weight_": 0.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.5, 0.0, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0, 0.0]],
            },
            "AUSTRIA": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 0.0,
                "cum_squtility_": 0.0,
                "cum_weight_": 0.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.5, 0.0, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0, 0.0]],
            },
            "GERMANY": {
                "single_power_cfrstats_version_": 2,
                "use_linear_weighting_": True,
                "use_optimistic_cfr_": False,
                "qre_": False,
                "qre_target_blueprint_": True,
                "qre_eta_": 2.0,
                "cum_utility_": 0.0,
                "cum_squtility_": 0.0,
                "cum_weight_": 0.0,
                "qre_lambda_": 3.0,
                "qre_entropy_factor_": 0.6,
                "actions_": [[0.6, 0.5, 0.0, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0, 0.0]],
            },
        }

    def test_single(self):
        use_linear_weighting = True
        cfr_optimistic = False
        qre = False
        qre_target_blueprint = True
        qre_eta = 2.0
        qre_lambda = 3.0
        qre_entropy_factor = 0.6
        bp_action_relprobs = [0.6, 0.4]
        stats = SinglePowerCFRStats(
            use_linear_weighting,
            cfr_optimistic,
            qre,
            qre_target_blueprint,
            qre_eta,
            qre_lambda,
            qre_entropy_factor,
            bp_action_relprobs,
        )
        stats.update(4.5, [1.0, 2.0], CFRStats.ACCUMULATE_PREV_ITER, 0)
        stats.update(10.5, [10.0, 20.0], CFRStats.ACCUMULATE_PREV_ITER, 1)
        dumped = pickle.dumps(stats)
        loaded = pickle.loads(dumped)

        assert loaded.bp_strategy(1.0) == [0.6, 0.4]
        assert loaded.bp_strategy(1.0) == [0.6, 0.4]
        assert abs(loaded.avg_utility() - 8.5) <= 0.001
        assert abs(loaded.avg_action_utilities()[0] - 7.0) <= 0.001
        assert abs(loaded.avg_action_utilities()[1] - 14.0) <= 0.001

        assert loaded.__getstate__() == {
            "single_power_cfrstats_version_": 2,
            "use_linear_weighting_": True,
            "use_optimistic_cfr_": False,
            "qre_": False,
            "qre_target_blueprint_": True,
            "qre_eta_": 2.0,
            "cum_utility_": 12.75000225,
            "cum_squtility_": 120.375010125,
            "cum_weight_": 1.5000005,
            "qre_lambda_": 3.0,
            "qre_entropy_factor_": 0.6,
            "actions_": [
                [0.6, 0.0, 0.25000025, -2.25000175, 10.5000005],
                [0.4, 1.0, 1.25000025, 8.24999875, 21.000001],
            ],
        }

    def test_old(self):
        # Make sure we can load old state too
        with open(os.path.dirname(__file__) + "/data/old_cfrstats.pickle", "rb") as f:
            loaded = pickle.load(f)

        assert loaded.__getstate__() == {
            "single_power_cfrstats_version_": 2,
            "use_linear_weighting_": True,
            "use_optimistic_cfr_": False,
            "qre_": False,
            "qre_target_blueprint_": True,
            "qre_eta_": 2.0,
            "cum_utility_": 12.75000225,
            "cum_squtility_": 120.375010125,
            "cum_weight_": 1.5000005,
            "qre_lambda_": 3.0,
            "qre_entropy_factor_": 1.0,
            "actions_": [
                [0.6, 0.0, 0.25000025, -2.25000175, 10.5000005],
                [0.4, 1.0, 1.25000025, 8.24999875, 21.000001],
            ],
        }
