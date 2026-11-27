"""Check the four scoring formulas and daylight selection."""

import unittest

import numpy as np

from abqcr import score


class TestScores(unittest.TestCase):
    def test_all_four_metrics_against_direct_formulas(self):
        first = np.linspace(0.1, 0.9, 99)
        second = np.repeat(0.7, 99)
        forecasts = np.stack((first, second, np.repeat(0.3, 99)))
        obs = np.array([0.5, 0.7, 0.8])
        got = score(forecasts, obs, daylight=np.array([True, True, False]))
        self.assertEqual(set(got), {"QS", "CRPS", "CE", "MAE", "records"})
        q, y = forecasts[:2], obs[:2, None]
        tau = np.arange(1, 100) / 100
        residual = y - q
        pinball = np.where(residual >= 0, residual * tau, residual * (tau - 1))
        crps = np.abs(residual).mean() - 0.5 * np.abs(q[:, :, None] - q[:, None, :]).mean()
        coverage_lo, coverage_hi = (y < q).mean(axis=0), (y <= q).mean(axis=0)
        projected_coverage = np.clip(tau, coverage_lo, coverage_hi)
        self.assertAlmostEqual(got["QS"], pinball.mean())
        self.assertAlmostEqual(got["CRPS"], crps)
        self.assertAlmostEqual(got["CE"], 100 * np.abs(projected_coverage - tau).mean())
        self.assertAlmostEqual(got["MAE"], np.abs(residual[:, 49]).mean())
        self.assertEqual(got["records"], 2)

    def test_rejects_crossing_issued_curves(self):
        descending = np.linspace(1, 0, 99)[None, :]
        with self.assertRaises(ValueError):
            score(descending, [0.5])

    def test_all_forecasts_equal_observation(self):
        got = score(np.full((2, 99), 0.25), [0.25, 0.25])
        self.assertEqual(got["CE"], 0)
        self.assertEqual(got["CRPS"], 0)


if __name__ == "__main__":
    unittest.main()
