"""Check projection, layout size, and serialization on synthetic examples."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from abqcr import ABQCR, project_quantiles


class TestCalibration(unittest.TestCase):
    def test_pava_merges_before_bounding(self):
        raw = np.full((1, 99), 0.8)
        raw[0, :2] = (1.7, -0.3)
        raw[0, -1] = 1.5
        projected = project_quantiles(raw)
        np.testing.assert_allclose(projected[0, :2], (0.7, 0.7))
        np.testing.assert_array_equal(projected[0, 2:-1], raw[0, 2:-1])
        np.testing.assert_allclose(projected[0, -1], 1.0)

    def test_fit_save_and_reload_with_three_anchors(self):
        rng = np.random.default_rng(18)
        raw = np.clip(rng.uniform(.1, .5, (12, 1)) + np.linspace(-.2, .3, 99), 0, 1)
        leads = np.tile([720, 915, 1065, 2000], 3)
        observations = raw[:, 49] + 0.07
        model = ABQCR.fit(raw, leads, observations, anchors=(1, 50, 99), updates=64)
        self.assertEqual(model.parameter_count, 468)
        self.assertLessEqual(model.fit_summary["saved_J"], model.fit_summary["initial_J"])
        before = model.predict(raw, leads)
        self.assertTrue(np.all(np.diff(before, axis=1) >= 0))
        self.assertTrue(np.all((before >= 0) & (before <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrator.npz"
            model.save(path)
            restored = ABQCR.load(path)
            np.testing.assert_array_equal(before, restored.predict(raw, leads))
            self.assertEqual(restored.fit_summary, model.fit_summary)

    def test_zero_updates_reproduce_the_monotone_baseline(self):
        raw = np.zeros((3, 99))
        raw[0, 44] = .5
        raw[1, 71] = .7
        leads = np.full(3, 720)
        model = ABQCR.fit(raw, leads, np.zeros(3), updates=0)
        self.assertEqual(model.parameter_count, 888)
        np.testing.assert_array_equal(model.predict(raw, leads), project_quantiles(raw))


if __name__ == "__main__":
    unittest.main()
