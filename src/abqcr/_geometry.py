"""Float64 geometry and linear operators for the full anchor–bridge model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


N_QUANTILES = 99
TAU = np.arange(1, N_QUANTILES + 1, dtype=np.float64) / 100
REFERENCE_ANCHORS = (1, 10, 50, 90, 99)
_FRI_INDEX = np.array(REFERENCE_ANCHORS) - 1
_EPSILON = 1e-4
_CUBIC_KNOTS = (0, 0, 0, 0, .2, .4, .6, .8, 1, 1, 1, 1)
_N_FEATURES = 6
_N_STATES = 27
_N_SPLINES = 6


def float_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def quantile_matrix(quantiles):
    array = float_array(quantiles, "quantiles")
    if array.ndim != 2 or array.shape[1] != N_QUANTILES:
        raise ValueError("quantiles must have shape (n_records, 99) in q01 to q99 order")
    return array


def lead_vector(lead_time_min, n_records):
    lead = float_array(lead_time_min, "lead_time_min")
    if lead.shape != (n_records,):
        raise ValueError("lead_time_min must contain one value per row of quantiles")
    return lead


def parse_anchors(anchors):
    values = float_array(anchors, "anchors")
    if values.ndim != 1 or len(values) < 2 or any(isinstance(value, (bool, np.bool_)) for value in anchors):
        raise ValueError("anchors must contain at least two positions")
    if np.all((0 < values) & (values < 1)):
        values = 100 * values
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0, atol=1e-12):
        raise ValueError("anchors must be q01 to q99 indices or corresponding probabilities")
    if rounded[0] != 1 or rounded[-1] != 99 or np.any(np.diff(rounded) < 7):
        raise ValueError("anchors must span q01 to q99 with successive gaps of at least 7 indices")
    return tuple(int(value) for value in rounded)


def nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def bounded_pava(values, *, merge_means=False):
    """Project each row by equal-weight PAVA, then bound each fitted level."""
    result = np.clip(values, 0., 1.)
    # Check the unbounded rows. Clipping before pooling would change the projection.
    crossed = np.flatnonzero(np.any(values[:, :-1] > values[:, 1:], axis=1))
    for index in crossed:
        blocks, lengths = [], []
        for value in values[index].tolist():
            blocks.append(value)
            lengths.append(1)
            while len(blocks) > 1:
                left, right = blocks[-2], blocks[-1]
                n_left, n_right = lengths[-2], lengths[-1]
                if merge_means:
                    if left <= right:
                        break
                    pooled = (left * n_left + right * n_right) / (n_left + n_right)
                else:
                    if left / n_left <= right / n_right:
                        break
                    pooled = left + right
                if not math.isfinite(pooled):
                    raise FloatingPointError("nonfinite pooled mean in bounded PAVA")
                blocks[-2] = pooled
                lengths[-2] = n_left + n_right
                blocks.pop()
                lengths.pop()
        if not merge_means:
            blocks = [total / length for total, length in zip(blocks, lengths)]
        result[index] = np.repeat(np.clip(blocks, 0., 1.), lengths)
    return result


@dataclass(frozen=True)
class _Geometry:
    anchors: tuple[int, ...]
    nodes: np.ndarray
    epsilon: float
    P: np.ndarray
    interiors: tuple[slice, ...]
    splines: tuple[np.ndarray, ...]
    curvature: np.ndarray
    state_penalty: np.ndarray
    lead_penalty: np.ndarray

    @property
    def theta_shape(self):
        return (_N_FEATURES * len(self.nodes), len(self.anchors))

    @property
    def bridge_shape(self):
        return (len(self.interiors), _N_STATES, _N_SPLINES)

    @property
    def main_denominator(self):
        return (len(self.nodes) - 2) * _N_FEATURES * len(self.anchors)

    @property
    def bridge_denominator(self):
        return len(self.interiors) * _N_STATES * _N_SPLINES


def geometry(anchors, time_knots, lead_bounds, epsilon):
    anchors = parse_anchors(anchors)
    time_knots = nonnegative_integer(time_knots, "time_knots")
    lead_bounds = float_array(lead_bounds, "lead_bounds")
    epsilon = float_array(epsilon, "epsilon")
    if time_knots < 3 or lead_bounds.shape != (2,) or lead_bounds[0] >= lead_bounds[1]:
        raise ValueError("use at least three time nodes and two increasing lead bounds")
    if epsilon.ndim != 0 or epsilon <= 0:
        raise ValueError("epsilon must be a positive scalar")
    nodes = np.linspace(*lead_bounds, time_knots)
    if not np.isfinite(nodes).all() or not np.all(nodes[1:] > nodes[:-1]):
        raise ValueError("lead bounds and time nodes must define distinct finite locations")

    P = np.zeros((N_QUANTILES, len(anchors)), dtype=np.float64)
    spline = BSpline(_CUBIC_KNOTS, np.eye(8), 3, extrapolate=False)
    interiors, splines = [], []
    for b, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        left, right = left - 1, right - 1
        t = np.arange(right - left + 1, dtype=np.float64) / (right - left)
        P[left:right + 1, b] = 1 - t
        P[left:right + 1, b + 1] = t
        interiors.append(slice(left + 1, right))
        # Drop the endpoint splines so that each bridge vanishes at both anchors.
        splines.append(spline(t[1:-1])[:, 1:7])

    roots, weights = np.polynomial.legendre.leggauss(4)
    curvature = np.zeros((_N_SPLINES, _N_SPLINES))
    second_derivative = spline.derivative(2)
    for left, right in zip(np.arange(5) / 5, np.arange(1, 6) / 5):
        points = (left + right) / 2 + roots * (right - left) / 2
        second = second_derivative(points)[:, 1:7]
        curvature += second.T @ ((weights * (right - left) / 2)[:, None] * second)
    curvature = _N_SPLINES * curvature / np.trace(curvature)

    difference = np.array([1., -2., 1.])
    gram, eye = np.outer(difference, difference), np.eye(3)
    state_penalty = (
        np.kron(np.kron(gram, eye), eye)
        + np.kron(np.kron(eye, gram), eye)
        + np.kron(np.kron(eye, eye), gram)
    )
    lead_difference = np.diff(np.eye(time_knots), n=2, axis=0)
    lead_penalty = lead_difference.T @ lead_difference
    return _Geometry(
        anchors, nodes, float(epsilon), P, tuple(interiors), tuple(splines),
        curvature, state_penalty, lead_penalty,
    )


def state_ranges(state_min, state_max, geom):
    lo = float_array(state_min, "state_min")
    hi = float_array(state_max, "state_max")
    if lo.shape != (len(geom.interiors), 3) or hi.shape != lo.shape or np.any(hi < lo):
        raise ValueError("state ranges must contain ordered level, log width, and lead bounds for each bridge")
    return lo, hi


class _Design:
    """Apply the main and bridge blocks without materializing their Jacobian."""

    def __init__(self, baseline, lead, geom, state=None):
        self.baseline = baseline
        self.geom = geom
        nodes = geom.nodes
        clamped = np.clip(lead, nodes[0], nodes[-1])
        hats = np.maximum(1 - np.abs(clamped[:, None] - nodes) / (nodes[1] - nodes[0]), 0)
        features = np.column_stack((np.ones(len(lead)), baseline[:, _FRI_INDEX]))
        self.chi = (hats[:, :, None] * features[:, None, :]).reshape(len(lead), geom.theta_shape[0])

        indices = np.asarray(geom.anchors) - 1
        low, high = baseline[:, indices[:-1]].T, baseline[:, indices[1:]].T
        level = (low + high) / 2
        log_width = np.log(high - low + geom.epsilon)
        local = np.stack((level, log_width, np.broadcast_to(lead / 60, level.shape)), axis=-1)
        lo, hi = (local.min(axis=1), local.max(axis=1)) if state is None else state
        self.state = (lo, hi)
        constant = hi == lo
        span = np.where(constant, 1., hi - lo)
        z = np.clip((local - lo[:, None, :]) / span[:, None, :], 0, 1)
        z = np.where(constant[:, None, :], .5, z)
        bern = np.stack(((1 - z) ** 2, 2 * z * (1 - z), z ** 2), axis=-1)
        self.phi = (
            bern[:, :, 0, :, None, None]
            * bern[:, :, 1, None, :, None]
            * bern[:, :, 2, None, None, :]
        ).reshape(len(geom.interiors), len(lead), _N_STATES)
        # FRI amplitude stays on q90–q10 when output anchor positions change.
        self.amplitude = baseline[:, 89] - baseline[:, 9] + geom.epsilon

    def parts(self, theta, bridge):
        main = (self.chi @ theta) @ self.geom.P.T
        detail = np.zeros_like(main)
        for b, (interior, spline) in enumerate(zip(self.geom.interiors, self.geom.splines)):
            detail[:, interior] = self.amplitude[:, None] * ((self.phi[b] @ bridge[b]) @ spline.T)
        return main, detail

    def forward(self, theta, bridge):
        main, detail = self.parts(theta, bridge)
        return main + detail

    def predict(self, theta, bridge):
        main, detail = self.parts(theta, bridge)
        return self.baseline + main + detail

    def adjoint(self, residual):
        main = self.chi.T @ (residual @ self.geom.P)
        detail = [
            self.phi[b].T @ ((self.amplitude[:, None] * residual[:, interior]) @ spline)
            for b, (interior, spline) in enumerate(zip(self.geom.interiors, self.geom.splines))
        ]
        return main, np.stack(detail)
