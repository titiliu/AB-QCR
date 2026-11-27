"""Pilot regressions and layout selection for 99 quantiles."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral

import numpy as np
from scipy.optimize import OptimizeWarning, linprog


THREE_ANCHORS = (1, 50, 99)
FIXED_FIVE_ANCHORS = (1, 10, 50, 90, 99)
LEFT_CANDIDATES = tuple(range(8, 44))
RIGHT_CANDIDATES = tuple(range(57, 93))
POLICIES = ("P00", "P10", "P01", "P11")
LAYOUT_METRICS = ("QS", "CRPS", "CE", "MAE")

_LP_OPTIONS = {
    "presolve": True,
    "dual_feasibility_tolerance": 1e-7,
    "primal_feasibility_tolerance": 1e-7,
    "ipm_optimality_tolerance": 1e-8,
    "parallel": False,
}


@dataclass(frozen=True)
class PilotResult:
    """Independent per-unit regressions and the weighted correction Gram matrix."""

    alpha: np.ndarray
    beta: np.ndarray
    weighted_pinball: np.ndarray
    correction_gram: np.ndarray


@dataclass(frozen=True)
class AnchorProposal:
    """Proposed output positions and the interpolation errors on each side."""

    left: int
    right: int
    left_errors: dict[int, float]
    right_errors: dict[int, float]
    representation_error: float

    @property
    def anchors(self) -> tuple[int, int, int, int, int]:
        return (1, self.left, 50, self.right, 99)


@dataclass(frozen=True)
class PolicySelection:
    """Winning side policy, its full-CAL update target, and all policy scores."""

    policy: str
    target_updates: int
    scores: dict[str, float]


@dataclass(frozen=True)
class LayoutSelection:
    """Competition ranks and layouts tied at the best mean rank and QS."""

    layout: str | None
    ranks: dict[str, dict[str, int]]
    mean_ranks: dict[str, float]
    denominator: int
    ties: tuple[str, ...]


def _fit_head(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float
) -> tuple[float, float, float]:
    if np.all(x == x[0]):
        residual = y - x
        order = np.argsort(residual, kind="stable")
        cumulative = np.cumsum(weights[order], dtype=np.float64)
        index = min(int(np.searchsorted(cumulative, tau * weights.sum())), len(y) - 1)
        alpha, beta = float(residual[order[index]]), 1.0
    else:
        lo, hi = float(x.min()), float(x.max())
        scale = hi - lo
        center = lo + scale / 2
        design = np.vstack((np.ones(len(x)), (x - center) / scale))
        bounds = np.column_stack(((tau - 1) * weights, tau * weights))
        # Request serial solving without changing HiGHS' process-wide scheduler.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected.*", category=OptimizeWarning
            )
            result = linprog(
                -y, A_eq=design, b_eq=np.zeros(2), bounds=bounds,
                method="highs-ipm", options=dict(_LP_OPTIONS),
            )
        marginals = getattr(getattr(result, "eqlin", None), "marginals", None)
        if (
            not result.success or marginals is None or np.shape(marginals) != (2,)
            or not np.isfinite(marginals).all()
        ):
            raise RuntimeError(f"Pilot regression failed at q{round(100 * tau):02d}. {result.message}")
        scaled_intercept, scaled_slope = -marginals
        beta = float(scaled_slope / scale)
        alpha = float(scaled_intercept - beta * center)

    residual = y - alpha - beta * x
    pinball = float(weights @ np.maximum(tau * residual, (tau - 1) * residual))
    if not all(map(math.isfinite, (alpha, beta, pinball))):
        raise FloatingPointError(f"Nonfinite pilot fit at q{round(100 * tau):02d}")
    return alpha, beta, pinball


def fit_pilot(
    m: np.ndarray, y: np.ndarray, lead_minutes: np.ndarray, *, capacity: float = 1.0
) -> PilotResult:
    """Fit independent pilot heads on the current legal fitting records.

    Supply an already projected N-by-99 baseline and a vector of observations in
    the same power units as capacity. Use all eligible fitting rows. Each
    distinct lead-time group receives equal total weight. Coefficients and losses
    are in per-unit coordinates, with predicted power given by
    `capacity * alpha + beta * m`.
    """
    m = np.asarray(m, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lead_minutes = np.asarray(lead_minutes, dtype=np.float64)
    capacity = float(capacity)
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Capacity must be finite and positive")
    if (
        m.ndim != 2 or m.shape[1] != 99 or not len(m)
        or y.shape != (len(m),) or lead_minutes.shape != (len(m),)
        or not all(np.isfinite(a).all() for a in (m, y, lead_minutes))
        or (m < 0).any() or (m > capacity).any()
        or (m[:, 1:] < m[:, :-1]).any()
    ):
        raise ValueError("Expected aligned finite rows and a bounded, ordered q01–q99 baseline")

    _, inverse, counts = np.unique(lead_minutes, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(counts) * counts[inverse])
    y_pu = y / capacity
    if not np.isfinite(y_pu).all():
        raise ValueError("Per-unit observations must be finite")
    fits = [_fit_head(m[:, j] / capacity, y_pu, weights, (j + 1) / 100) for j in range(99)]
    alpha, beta, losses = np.asarray(fits).T

    gram = np.zeros((99, 99), dtype=np.float64)
    for start in range(0, len(m), 4096):
        rows = slice(start, start + 4096)
        correction = alpha[None, :] + (beta[None, :] - 1.0) * (m[rows] / capacity)
        gram += correction.T @ (weights[rows, None] * correction)
    if not np.isfinite(gram).all():
        raise FloatingPointError("Nonfinite weighted pilot correction Gram matrix")
    return PilotResult(alpha, beta, losses, (gram + gram.T) / 2)


def _interval_error(gram: np.ndarray, first: int, last: int) -> float:
    # Only the interior terms contribute since the interpolation is exact at anchors.
    left, right = first - 1, last - 1
    j = np.arange(left + 1, right)
    t = (j - left) / (right - left)
    u = 1 - t
    terms = np.vstack((
        gram[j, j], u * u * gram[left, left], t * t * gram[right, right],
        2 * u * t * gram[left, right], -2 * u * gram[j, left], -2 * t * gram[j, right],
    ))
    value = math.fsum(terms.ravel()) / 99
    tolerance = 128 * np.finfo(np.float64).eps * math.fsum(np.abs(terms).ravel()) / 99
    if value < -tolerance:
        raise FloatingPointError(f"Materially negative Gram interpolation error {value}")
    return max(0.0, value)


def propose_layouts(pilot: PilotResult) -> AnchorProposal:
    """Minimize the weighted representation error over each side's search grid."""
    gram = np.asarray(pilot.correction_gram, dtype=np.float64)
    if gram.shape != (99, 99) or not np.isfinite(gram).all():
        raise ValueError("Expected a finite 99-by-99 weighted correction Gram matrix")
    left_errors = {
        a: _interval_error(gram, 1, a) + _interval_error(gram, a, 50)
        for a in LEFT_CANDIDATES
    }
    right_errors = {
        b: _interval_error(gram, 50, b) + _interval_error(gram, b, 99)
        for b in RIGHT_CANDIDATES
    }
    left = min(left_errors, key=lambda a: (left_errors[a], a))
    right = min(right_errors, key=lambda b: (right_errors[b], b))
    error = math.fsum((
        _interval_error(gram, 1, left), _interval_error(gram, left, 50),
        _interval_error(gram, 50, right), _interval_error(gram, right, 99),
    ))
    return AnchorProposal(left, right, left_errors, right_errors, error)


def policy_layouts(proposal: AnchorProposal) -> dict[str, tuple[int, ...]]:
    """Apply the four side policies to one fitting window's proposal."""
    return {
        "P00": FIXED_FIVE_ANCHORS,
        "P10": (1, proposal.left, 50, 90, 99),
        "P01": (1, 10, 50, proposal.right, 99),
        "P11": proposal.anchors,
    }


def select_policy(
    warm_qs: Mapping[str, float],
    cold_qs: Mapping[str, float],
    warm_updates: Mapping[str, int],
    cold_updates: Mapping[str, int],
) -> PolicySelection:
    """Select from four policies' eligible warm- and cold-DEV checkpoints.

    Supply daytime QS of the issued curves and the corresponding positive update
    counts. Use the lower update count when checkpoints tie on QS. Apply the
    chosen policy to the full-CAL proposal and use the returned update target.
    P00 uses the fixed anchors directly.
    """
    expected = set(POLICIES)
    if any(set(values) != expected for values in (warm_qs, cold_qs, warm_updates, cold_updates)):
        raise ValueError("Supply scores and update counts for P00, P10, P01, and P11 in both windows")
    warm = {policy: float(warm_qs[policy]) for policy in POLICIES}
    cold = {policy: float(cold_qs[policy]) for policy in POLICIES}
    if any(not math.isfinite(qs) or qs < 0 for qs in (*warm.values(), *cold.values())):
        raise ValueError("DEV QS must be finite and nonnegative")
    if warm["P00"] <= 0 or cold["P00"] <= 0:
        raise ValueError("Both P00 scores must be positive to normalize policy scores")
    if any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1
        for counts in (warm_updates, cold_updates) for n in counts.values()
    ):
        raise ValueError("An eligible checkpoint requires a positive integer update count")

    scores = {
        p: 0.5 * (warm[p] / warm["P00"] + cold[p] / cold["P00"]) for p in POLICIES
    }
    if not all(map(math.isfinite, scores.values())):
        raise ValueError("Normalized policy scores must be finite")
    winner = min(POLICIES, key=lambda p: (scores[p], p.count("1")))
    target = (int(warm_updates[winner]) + int(cold_updates[winner]) + 1) // 2
    return PolicySelection(winner, target, scores)


def _metric_values(scores: Mapping[str, float | str | Decimal]) -> tuple[Decimal, ...]:
    try:
        values = tuple(Decimal(str(scores[metric])) for metric in LAYOUT_METRICS)
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise ValueError("Supply unrounded QS, CRPS, CE, and MAE") from exc
    if not all(value.is_finite() for value in values):
        raise ValueError("Layout comparison requires finite scores for all four metrics")
    return values


def select_layout(
    candidate_scores: Mapping[str, Mapping[str, float | str | Decimal]],
    comparator_scores: Mapping[str, Mapping[str, float | str | Decimal]],
) -> LayoutSelection:
    """Compare each layout independently against the same static comparators.

    Supply unrounded scores on the same development validation rows and units.
    Metric ranks count only strictly better comparators. The four ranks receive
    equal weight and lower QS breaks mean-rank ties. If both criteria still tie,
    `layout` is None and `ties` lists the tied names.
    """
    if not candidate_scores or not comparator_scores:
        raise ValueError("Supply at least one candidate and one comparator")
    if candidate_scores.keys() & comparator_scores.keys():
        raise ValueError("Candidate and comparator names must be distinct")

    candidates = {name: _metric_values(scores) for name, scores in candidate_scores.items()}
    comparators = tuple(_metric_values(scores) for scores in comparator_scores.values())
    ranks = {
        name: {
            metric: 1 + sum(peer[j] < values[j] for peer in comparators)
            for j, metric in enumerate(LAYOUT_METRICS)
        }
        for name, values in candidates.items()
    }
    rank_sums = {name: sum(positions.values()) for name, positions in ranks.items()}
    keys = {name: (rank_sums[name], values[0]) for name, values in candidates.items()}
    best = min(keys.values())
    ties = tuple(name for name, key in keys.items() if key == best)
    means = {name: total / len(LAYOUT_METRICS) for name, total in rank_sums.items()}
    return LayoutSelection(ties[0] if len(ties) == 1 else None, ranks, means, len(comparators) + 1, ties)
