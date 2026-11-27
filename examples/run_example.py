"""Run the CAL workflow on forecasts generated in memory."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from abqcr import ABQCR, CALSplits, fit_cal, project_quantiles


def make_forecasts(day_count: int, seed: int):
    rng = np.random.default_rng(seed)
    lead = np.linspace(720, 2145, 10).round().astype(int)
    hour = (lead - 720) / 60
    profile = np.maximum(np.sin(np.pi * (hour - 6) / 13), 0.0) ** 1.5
    cloud = rng.uniform(0.5, 0.95, (day_count, 1))
    observed = np.clip(
        profile * (cloud + rng.normal(0, 0.05, (day_count, len(lead)))), 0, 1
    )
    level = profile * (0.85 * cloud + 0.11 + rng.normal(0, 0.03, observed.shape))
    spread = 0.012 + 0.12 * profile
    probabilities = np.arange(1, 100) / 100
    quantiles = level[:, :, None] + spread[None, :, None] * (2 * probabilities - 1)
    quantiles[::2, :, 40:43] = quantiles[::2, :, 40:43][:, :, ::-1].copy()
    daylight = (hour > 6) & (hour < 19)
    return (
        quantiles.reshape(-1, 99),
        np.tile(lead, day_count),
        observed.ravel(),
        np.tile(daylight, day_count),
    )


def main() -> None:
    quantiles, lead, observed, daylight = make_forecasts(day_count=14, seed=177)
    rows = np.arange(len(lead)).reshape(14, -1)
    splits = CALSplits(
        train=rows[:9].ravel(),
        stop=rows[9:11].ravel(),
        select=rows[11:].ravel(),
        warm_fit=rows[:5].ravel(),
        warm_dev=rows[5:8].ravel(),
        cold_fit=rows[:8].ravel(),
    )
    comparators = {"MONOTONE": project_quantiles(quantiles[splits.select])}
    model, report = fit_cal(
        quantiles,
        lead,
        observed,
        daylight,
        splits,
        comparators,
        rungs=((8, 9), (24, 3)),
        checkpoint_updates=(8, 16, 24),
    )
    future, future_lead, _, _ = make_forecasts(day_count=3, seed=178)
    issued = model.predict(future, future_lead)
    if issued.shape != (len(future_lead), 99):
        raise AssertionError("Expected 99 quantiles for each forecast record.")
    if np.any(np.diff(issued, axis=1) < 0) or np.any((issued < 0) | (issued > 1)):
        raise AssertionError("Expected noncrossing forecasts in [0, 1].")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "fitted.npz"
        model.save(path)
        restored = ABQCR.load(path)
        np.testing.assert_array_equal(restored.predict(future, future_lead), issued)

    print(json.dumps({
        "selected_layout": report["selected_layout"],
        "anchors": list(model.anchors),
        "coefficients": model.parameter_count,
        "forecast_rows": len(issued),
        "quantiles_per_forecast": issued.shape[1],
        "cal_saved_updates": model.fit_summary["saved_updates"],
    }, indent=2))


if __name__ == "__main__":
    main()
