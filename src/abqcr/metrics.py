"""Daylight scores for final issued curves on the q01–q99 grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

_PROBABILITIES = np.arange(1, 100, dtype=np.float64) / 100
_CRPS_COEFFICIENTS = 2 * np.arange(1, 100) - 100


def score(
    quantiles: ArrayLike, observed_pu: ArrayLike, *, daylight: ArrayLike | None = None
) -> dict[str, float | int]:
    """Return QS, CRPS, CE (percentage points), MAE, and the record count.

    The power-based scores use per-unit values. Provide the verified daylight
    mask for the paper's evaluation. With no mask, score all supplied rows.
    """
    values = np.asarray(quantiles, dtype=np.float64)
    observations = np.asarray(observed_pu, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 99 or observations.shape != (len(values),):
        raise ValueError("Expected quantiles of shape (n, 99) and observations of shape (n,).")
    if daylight is not None:
        mask = np.asarray(daylight)
        if mask.shape != (len(values),) or mask.dtype != np.dtype(bool):
            raise ValueError("daylight must be a Boolean vector of length n.")
        values, observations = values[mask], observations[mask]
    if not len(values):
        raise ValueError("Evaluation requires at least one eligible record.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(observations)):
        raise ValueError("Eligible predictions and observations must be finite.")
    if np.any(np.diff(values, axis=1) < 0) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Evaluate issued curves with ordered quantiles in [0, 1].")

    residual = observations[:, None] - values
    quantile_losses = np.maximum(
        _PROBABILITIES * residual, (_PROBABILITIES - 1) * residual
    ).mean(axis=0, dtype=np.float64)
    strict_coverage = (residual < 0).mean(axis=0)
    inclusive_coverage = (residual <= 0).mean(axis=0)
    coverage_error = np.maximum(
        np.maximum(strict_coverage - _PROBABILITIES, _PROBABILITIES - inclusive_coverage),
        0,
    )

    crps = np.abs(residual).mean() - (values @ _CRPS_COEFFICIENTS).mean() / 99**2
    return {
        "QS": float(quantile_losses.mean()),
        "CRPS": float(crps),
        "CE": float(100 * coverage_error.mean()),
        "MAE": float(np.abs(residual[:, 49]).mean()),
        "records": len(values),
    }
