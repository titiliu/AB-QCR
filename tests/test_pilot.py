import unittest

import numpy as np
from scipy.optimize import linprog

from abqcr.pilot import (
    LAYOUT_METRICS,
    POLICIES,
    PilotResult,
    fit_pilot,
    policy_layouts,
    propose_layouts,
    select_layout,
    select_policy,
)


class PilotTests(unittest.TestCase):
    def test_constant_columns_use_equal_lead_mass(self):
        m = np.zeros((4, 99))
        result = fit_pilot(m, [0.1, 0.2, 0.3, 0.8], [720, 720, 720, 735])
        self.assertEqual(result.alpha[49], 0.3)
        self.assertEqual(result.alpha[50], 0.8)
        np.testing.assert_array_equal(result.beta, np.ones(99))
        np.testing.assert_allclose(result.correction_gram, np.outer(result.alpha, result.alpha))

    def test_nonconstant_dual_lp_has_free_slope(self):
        # The pilot also works after the process has used the default HiGHS scheduler.
        self.assertTrue(linprog([1], bounds=(0, None), method="highs").success)
        x = np.linspace(0.1, 0.5, 8)
        offsets = np.arange(99) / 1000
        m = x[:, None] + offsets
        leads = [720] * 5 + [735] * 3
        result = fit_pilot(128 * m, 128 * (0.8 - 0.5 * x), leads, capacity=128)
        np.testing.assert_allclose(result.alpha, 0.8 + 0.5 * offsets, atol=1e-13)
        np.testing.assert_allclose(result.beta, -0.5, atol=1e-13)
        np.testing.assert_allclose(result.weighted_pinball, 0, atol=1e-14)

    def test_gram_representation_selects_the_known_knots(self):
        anchors = (1, 21, 50, 76, 99)
        grid = np.arange(1, 100)
        correction = np.array([
            np.interp(grid, anchors, [0.2, 0.7, -0.3, 0.4, 0.1]),
            np.interp(grid, anchors, [0.6, -0.1, -0.5, 0.15, -0.25]),
        ])
        weights = np.array([0.25, 0.75])
        gram = correction.T @ (weights[:, None] * correction)
        result = propose_layouts(PilotResult(np.zeros(99), np.ones(99), np.zeros(99), gram))
        self.assertEqual(result.anchors, anchors)
        self.assertLess(result.representation_error, 1e-15)
        self.assertEqual(policy_layouts(result), {
            "P00": (1, 10, 50, 90, 99),
            "P10": (1, 21, 50, 90, 99),
            "P01": (1, 10, 50, 76, 99),
            "P11": anchors,
        })
        zero = fit_pilot(np.zeros((2, 99)), [0, 0], [720, 735])
        self.assertEqual(propose_layouts(zero).anchors, (1, 8, 50, 57, 99))

    def test_policy_selection_normalizes_by_window_and_breaks_ties(self):
        warm = dict(zip(POLICIES, (10, 8, 9, 8)))
        cold = dict(zip(POLICIES, (20, 20, 18, 20)))
        warm_updates = dict.fromkeys(POLICIES, 256)
        cold_updates = dict.fromkeys(POLICIES, 513)
        result = select_policy(warm, cold, warm_updates, cold_updates)
        self.assertEqual(result.policy, "P10")
        self.assertEqual(result.target_updates, 385)
        self.assertEqual(result.scores["P01"], result.scores["P10"])
        self.assertEqual(result.scores["P11"], result.scores["P10"])

    def test_layout_rank_preserves_unrounded_scores_and_ignores_other_candidates(self):
        peer = dict(zip(LAYOUT_METRICS, ("1.0000000000000000002", "2", "3", "4", "5")))
        left = {**peer, "QS": "1.0000000000000000001", "CRPS": "2.1"}
        right = {**peer, "QS": "1.0000000000000000003", "CRPS": "1.9"}
        result = select_layout({"left": left, "right": right}, {"static": peer})
        self.assertEqual(result.denominator, 2)
        self.assertEqual(result.ranks["left"], dict(zip(LAYOUT_METRICS, (1, 2, 1, 1, 1))))
        self.assertEqual(result.ranks["right"], dict(zip(LAYOUT_METRICS, (2, 1, 1, 1, 1))))
        self.assertEqual(result.layout, "left")
        tied = select_layout({"left": left, "duplicate": left}, {"static": peer})
        self.assertIsNone(tied.layout)
        self.assertEqual(tied.ties, ("left", "duplicate"))


if __name__ == "__main__":
    unittest.main()
