"""Public fit, predict and serialization API for normalized AB-QCR."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._geometry import (
    REFERENCE_ANCHORS,
    _Design,
    _EPSILON,
    _Geometry,
    bounded_pava,
    float_array,
    geometry,
    lead_vector,
    nonnegative_integer,
    quantile_matrix,
    state_ranges,
)
from ._solver import horizon_weights, solve


_MODEL_SCHEMA = "abqcr-npz-v1"
_MODEL_MEMBERS = {"theta", "bridge", "state_min", "state_max", "metadata"}
_DEFAULT_LEAD_BOUNDS = (720, 2145)


def project_quantiles(quantiles):
    """Return the bounded Euclidean PAVA reference for complete q01–q99 rows."""
    return bounded_pava(quantile_matrix(quantiles))


def _lambdas(regularization):
    values = float_array(regularization, "regularization")
    if values.shape != (3,) or np.any(values < 0):
        raise ValueError("regularization must contain three nonnegative values (lambda_h, lambda_s, lambda_q)")
    return tuple(values.tolist())


def _readonly(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class ABQCR:
    """Fitted full anchor–bridge correction for quantiles in per-unit capacity."""

    _geometry: _Geometry = field(repr=False)
    theta: np.ndarray = field(repr=False)
    bridge: np.ndarray = field(repr=False)
    state_min: np.ndarray = field(repr=False)
    state_max: np.ndarray = field(repr=False)
    regularization: tuple[float, float, float]
    fit_summary: dict | None = field(default=None, repr=False)

    @property
    def anchors(self):
        """One-based output anchor indices on the q01–q99 probability grid."""
        return self._geometry.anchors

    @property
    def time_knots(self):
        return len(self._geometry.nodes)

    @property
    def lead_bounds(self):
        return float(self._geometry.nodes[0]), float(self._geometry.nodes[-1])

    @property
    def epsilon(self):
        return self._geometry.epsilon

    @property
    def parameter_count(self):
        return self.theta.size + self.bridge.size

    @classmethod
    def from_coefficients(
        cls,
        *,
        theta,
        bridge,
        state_min,
        state_max,
        anchors=REFERENCE_ANCHORS,
        time_knots=8,
        lead_bounds=_DEFAULT_LEAD_BOUNDS,
        epsilon=_EPSILON,
        regularization=(0., 0., 0.),
    ):
        """Construct a model from externally frozen float64 coefficients.

        The main matrix has shape (6 * time_knots, n_anchors). The bridge
        tensor has shape (n_anchors - 1, 27, 6). Each state's two range arrays
        have shape (n_anchors - 1, 3), ordered level, log width and lead hours.
        """
        geom = geometry(anchors, time_knots, lead_bounds, epsilon)
        theta = float_array(theta, "theta")
        bridge = float_array(bridge, "bridge")
        if theta.shape != geom.theta_shape or bridge.shape != geom.bridge_shape:
            raise ValueError(f"expected theta {geom.theta_shape} and bridge {geom.bridge_shape}")
        lo, hi = state_ranges(state_min, state_max, geom)
        return cls(geom, _readonly(theta), _readonly(bridge), _readonly(lo), _readonly(hi), _lambdas(regularization))

    @classmethod
    def fit(
        cls,
        quantiles,
        lead_time_min,
        observed,
        *,
        anchors=REFERENCE_ANCHORS,
        time_knots=8,
        regularization=(1e-4, 1e-4, 1e-4),
        updates=256,
        lead_bounds=_DEFAULT_LEAD_BOUNDS,
        epsilon=_EPSILON,
        checkpoint_updates=None,
        selection="best_objective",
    ):
        """Fit a specified layout and regularization to all supplied rows.

        Quantiles have shape (n_records, 99), with observations and lead times
        as aligned one-dimensional vectors. Quantiles and observations use
        per-unit capacity. Each distinct lead has the same total loss weight.
        Supply anchor positions as indices, for example (1, 10, 50, 90, 99),
        or as probabilities, for example (.01, .10, .50, .90, .99).

        The default CAL schedule inspects the objective at zero, every 256
        updates through 4096, every 1024 thereafter, and the update target.
        Specify checkpoint_updates to replace the interior inspection points.
        Selection is 'best_objective' (earliest minimum) or 'last' (the target).
        """
        geom = geometry(anchors, time_knots, lead_bounds, epsilon)
        regularization = _lambdas(regularization)
        updates = nonnegative_integer(updates, "updates")
        if selection not in ("best_objective", "last"):
            raise ValueError("selection must be 'best_objective' or 'last'")
        baseline = project_quantiles(quantiles)
        lead = lead_vector(lead_time_min, len(baseline))
        observed = float_array(observed, "observed")
        if not len(baseline) or observed.shape != (len(baseline),):
            raise ValueError("fit needs at least one row and one observed value per row")
        design = _Design(baseline, lead, geom)
        theta, bridge, summary = solve(
            design, observed, horizon_weights(lead), regularization, updates,
            checkpoint_updates, selection,
        )
        lo, hi = design.state
        return cls(
            geom, _readonly(theta), _readonly(bridge), _readonly(lo), _readonly(hi),
            regularization, summary,
        )

    def predict_prepublication(self, quantiles, lead_time_min):
        """Apply the frozen correction before the final bounded projection."""
        baseline = project_quantiles(quantiles)
        lead = lead_vector(lead_time_min, len(baseline))
        design = _Design(baseline, lead, self._geometry, (self.state_min, self.state_max))
        pre = design.predict(self.theta, self.bridge)
        if not np.isfinite(pre).all():
            raise FloatingPointError("nonfinite prepublication AB-QCR prediction")
        return pre

    def predict(self, quantiles, lead_time_min):
        """Return published float64 quantiles in q01–q99 order, bounded and monotone."""
        return bounded_pava(self.predict_prepublication(quantiles, lead_time_min), merge_means=True)

    def save(self, path):
        """Atomically write a compressed NPZ with numeric arrays and JSON metadata."""
        path = Path(path)
        metadata = json.dumps({
            "schema": _MODEL_SCHEMA,
            "anchors": self.anchors,
            "time_knots": self.time_knots,
            "lead_bounds": self.lead_bounds,
            "epsilon": self.epsilon,
            "regularization": self.regularization,
            "fit_summary": self.fit_summary,
        }, allow_nan=False)
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(
                    handle, theta=self.theta, bridge=self.bridge, state_min=self.state_min,
                    state_max=self.state_max, metadata=np.array(metadata),
                )
            except BaseException:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Load a portable NPZ model with pickle deserialization disabled."""
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _MODEL_MEMBERS:
                raise ValueError("invalid AB-QCR model contents")
            metadata = data["metadata"]
            if metadata.shape != () or metadata.dtype.kind != "U":
                raise ValueError("invalid AB-QCR model metadata")
            meta = json.loads(metadata.item())
            if not isinstance(meta, dict) or meta.get("schema") != _MODEL_SCHEMA:
                raise ValueError("unsupported AB-QCR model schema")
            model = cls.from_coefficients(
                theta=data["theta"], bridge=data["bridge"], state_min=data["state_min"],
                state_max=data["state_max"], anchors=meta["anchors"], time_knots=meta["time_knots"],
                lead_bounds=meta["lead_bounds"], epsilon=meta["epsilon"], regularization=meta["regularization"],
            )
        summary = meta["fit_summary"]
        if summary is not None and not isinstance(summary, dict):
            raise ValueError("invalid AB-QCR fit summary")
        return cls(
            model._geometry, model.theta, model.bridge, model.state_min, model.state_max,
            model.regularization, summary,
        )
