import json
import unittest
from dataclasses import replace

import numpy as np

from abqcr import ABQCR, CALSplits, fit_cal, project_quantiles, score
from abqcr.pilot import POLICIES, fit_pilot, policy_layouts, propose_layouts, select_policy


class CALDevelopmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.observed = rng.uniform(0.15, 0.85, 32)
        center = np.clip(self.observed + rng.normal(0.16, 0.05, 32), 0.15, 0.95)
        self.raw = center[:, None] + np.linspace(-0.1, 0.15, 99)
        self.lead = np.tile([720, 1200, 1800, 2145], 8)
        self.daylight = np.arange(32) % 4 > 0
        self.splits = CALSplits(
            np.arange(16), np.arange(16, 24), np.arange(24, 32),
            np.arange(8), np.arange(9, 16), np.arange(23),
        )
        comparison = project_quantiles(self.raw[self.splits.select])
        self.comparators = {"monotone": comparison}
        self.rungs = ((2, 9), (6, 3))
        self.checkpoints = (1, 2, 3, 4, 6)

    def fit(self, **overrides):
        options = {
            "quantiles": self.raw,
            "lead_time_min": self.lead,
            "observed": self.observed,
            "daylight": self.daylight,
            "splits": self.splits,
            "comparator_forecasts": self.comparators,
            "rungs": self.rungs,
            "checkpoint_updates": self.checkpoints,
        }
        return fit_cal(**(options | overrides))

    def test_search_continues_trajectories_and_refits_the_selected_layout(self):
        target_time = np.datetime64("2020-01-01T00:00") + np.arange(32) * np.timedelta64(2, "D")
        leads, observations = self.lead, self.observed
        model, report = self.fit(
            target_time=target_time,
            available_time=target_time + np.timedelta64(1, "h"),
            issue_time=target_time - leads.astype("timedelta64[m]"),
        )
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["calendar_boundaries_checked"])
        self.assertEqual(report["rank_denominator"], 2)
        comparators = report["comparator_scores"].values()
        metrics = ("QS", "CRPS", "CE", "MAE")
        self.assertTrue(all(set(peer) == {*metrics, "records"} for peer in comparators))
        expected_ranks = {}
        for name, branch in report["development"].items():
            scores = branch["select_scores"]
            self.assertEqual(set(scores), {*metrics, "records"})
            expected_ranks[name] = {
                metric: 1 + sum(peer[metric] < scores[metric] for peer in comparators)
                for metric in metrics
            }
        self.assertEqual(report["layout_ranks"], expected_ranks)
        expected_means = {name: sum(ranks.values()) / 4 for name, ranks in expected_ranks.items()}
        self.assertEqual(report["layout_mean_ranks"], expected_means)
        layout_order = ("anchor3", "fixed5", "dd5")
        selected = min(layout_order, key=lambda name: (
            expected_means[name], report["development"][name]["select_scores"]["QS"],
        ))
        self.assertEqual(report["selected_layout"], selected)
        train, stop, select = self.splits.train, self.splits.stop, self.splits.select
        for branch in report["development"].values():
            search = branch["search"]
            first, second = search["rungs"]
            self.assertEqual(len(first["trials"]), 27)
            self.assertEqual(len(first["promoted_regularization"]), 9)
            self.assertEqual(len(second["trials"]), 9)
            self.assertEqual(len(second["promoted_regularization"]), 3)
            self.assertEqual(len(search["select_finalists"]), 3)
            stop_updates = search["best_stop_updates"]
            refit = ABQCR.fit(
                self.raw[train], leads[train], observations[train], anchors=search["anchors"],
                regularization=search["regularization"], updates=stop_updates, selection="last",
            )
            for stage, rows in (("STOP", stop), ("SELECT", select)):
                with self.subTest(anchors=search["anchors"], stage=stage):
                    scored = score(
                        refit.predict(self.raw[rows], leads[rows]),
                        observations[rows], daylight=self.daylight[rows],
                    )
                    expected = search["best_stop_qs"] if stage == "STOP" else search["select_scores"]["QS"]
                    self.assertEqual(scored["QS"], expected)

        dd5 = report["development"]["dd5"]
        policy = select_policy(
            {name: dd5["warm"][name]["best_qs"] for name in POLICIES},
            {name: dd5["cold"][name]["best_qs"] for name in POLICIES},
            {name: dd5["warm"][name]["best_updates"] for name in POLICIES},
            {name: dd5["cold"][name]["best_updates"] for name in POLICIES},
        )
        self.assertEqual(dd5["policy"], policy.policy)
        self.assertEqual(dd5["policy"], "P10")
        self.assertEqual(dd5["target_updates"], policy.target_updates)
        if selected == "dd5":
            proposal = propose_layouts(fit_pilot(project_quantiles(self.raw), observations, leads))
            expected_anchors = policy_layouts(proposal)[policy.policy]
            self.assertNotEqual(dd5["cold"][policy.policy]["anchors"], expected_anchors)
        else:
            expected_anchors = report["development"][selected]["search"]["anchors"]
        self.assertEqual(model.anchors, expected_anchors)
        cold_fit = self.splits.cold_fit
        cold_checkpoint = ABQCR.fit(
            self.raw[cold_fit], leads[cold_fit], observations[cold_fit],
            anchors=dd5["cold"][policy.policy]["anchors"], regularization=dd5["regularization"],
            updates=dd5["cold"][policy.policy]["best_updates"], selection="last",
        )
        self.assertEqual(
            dd5["select_scores"],
            score(
                cold_checkpoint.predict(self.raw[select], leads[select]),
                observations[select], daylight=self.daylight[select],
            ),
        )
        target = report["full_cal"]["target_updates"]
        self.assertEqual(report["development"][selected]["target_updates"], target)
        refit = ABQCR.fit(
            self.raw, leads, observations, anchors=model.anchors,
            regularization=model.regularization, updates=target,
            checkpoint_updates=[n for n in self.checkpoints if n <= target],
        )
        np.testing.assert_array_equal(refit.theta, model.theta)
        np.testing.assert_array_equal(refit.bridge, model.bridge)
        self.assertEqual(report["full_cal"]["records"], len(self.raw))
        self.assertEqual(report["full_cal"]["saved_J"], refit.fit_summary["saved_J"])

    def test_split_and_timestamp_validation(self):
        with self.assertRaisesRegex(ValueError, "train must contain strictly increasing"):
            self.fit(splits=replace(self.splits, train=[0, 2, 2]))
        with self.assertRaisesRegex(ValueError, "train rows must strictly precede stop"):
            self.fit(splits=replace(self.splits, train=np.arange(17)))
        with self.assertRaisesRegex(ValueError, "cold_fit rows must strictly precede select"):
            self.fit(splits=replace(self.splits, cold_fit=np.arange(25)))
        with self.assertRaisesRegex(ValueError, "stop must contain at least one daylight"):
            daylight = self.daylight.copy()
            daylight[self.splits.stop] = False
            self.fit(daylight=daylight)
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            self.fit(target_time=np.arange(32))
        target = 10 * np.arange(32)
        available = target + 1
        issue = target - 2
        available[self.splits.train[-1]] = issue[self.splits.stop[0]]
        with self.assertRaisesRegex(ValueError, "earliest stop issuance"):
            self.fit(target_time=target, available_time=available, issue_time=issue)

    def test_comparator_covers_all_select_rows(self):
        values = self.comparators["monotone"]
        with self.assertRaisesRegex(ValueError, "cover every SELECT row"):
            self.fit(comparator_forecasts={"monotone": values[1:]})
        with self.assertRaisesRegex(ValueError, "bounded and ordered issued quantiles"):
            invalid = values.copy()
            invalid[0, 30] = 2.0
            self.fit(comparator_forecasts={"monotone": invalid})

    def test_zero_fit_does_not_supply_a_development_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "nonzero positive-update STOP"):
            self.fit(quantiles=np.zeros_like(self.raw), observed=np.zeros_like(self.observed))


if __name__ == "__main__":
    unittest.main()
