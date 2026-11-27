"""Select an AB-QCR layout and its fitting settings using CAL data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from ._geometry import (
    _Design,
    _EPSILON,
    bounded_pava,
    float_array,
    geometry,
    lead_vector,
    nonnegative_integer,
    quantile_matrix,
)
from ._solver import _ChambollePock, _saved_updates, horizon_weights
from .core import ABQCR, _DEFAULT_LEAD_BOUNDS, project_quantiles
from .metrics import score
from .pilot import (
    FIXED_FIVE_ANCHORS,
    POLICIES,
    THREE_ANCHORS,
    fit_pilot,
    policy_layouts,
    propose_layouts,
    select_layout,
    select_policy,
)


_LAYOUTS = ("anchor3", "fixed5", "dd5")
_REGULARIZATION_GRID = tuple(product((0.0, 1e-4, 1e-2), repeat=3))
_DEFAULT_RUNGS = ((4096, 9), (16384, 3), (65536, 3), (131072, 3))
_PATIENCE = 16


@dataclass(frozen=True)
class CALSplits:
    """Aligned zero-based indices into chronologically ordered CAL rows.

    TRAIN precedes STOP, which precedes SELECT. Each policy FIT window
    precedes its DEV window. Cold DEV uses exactly SELECT. A row can play
    different roles in different development windows.
    """

    train: ArrayLike
    stop: ArrayLike
    select: ArrayLike
    warm_fit: ArrayLike
    warm_dev: ArrayLike
    cold_fit: ArrayLike


_SPLIT_FIELDS = tuple(CALSplits.__dataclass_fields__)
_ORDERED_SPLITS = (
    ("train", "stop"), ("stop", "select"),
    ("warm_fit", "warm_dev"), ("cold_fit", "select"),
)


def _validate_splits(splits, n_records, daylight, timestamps):
    if not isinstance(splits, CALSplits):
        raise ValueError("splits must be a CALSplits instance")
    indices = {}
    for name in _SPLIT_FIELDS:
        rows = np.asarray(getattr(splits, name))
        if (
            rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu"
            or rows[0] < 0 or rows[-1] >= n_records or np.any(rows[1:] <= rows[:-1])
        ):
            raise ValueError(f"{name} must contain strictly increasing CAL row indices")
        indices[name] = rows.astype(np.intp, copy=False)
    for first, second in _ORDERED_SPLITS:
        if indices[first][-1] >= indices[second][0]:
            raise ValueError(f"{first} rows must strictly precede {second} rows")
    for name in ("stop", "select", "warm_dev"):
        if not np.any(daylight[indices[name]]):
            raise ValueError(f"{name} must contain at least one daylight record")
    if any(values is not None for values in timestamps):
        _validate_timestamps(timestamps, n_records, indices)
    return CALSplits(**indices)


def _validate_timestamps(timestamps, n_records, indices):
    names = ("target_time", "available_time", "issue_time")
    if any(values is None for values in timestamps):
        raise ValueError("target_time, available_time, and issue_time must be supplied together")
    target, available, issue = arrays = tuple(np.asarray(values) for values in timestamps)
    for name, values in zip(names, arrays):
        if (
            values.shape != (n_records,) or values.dtype.kind not in "Mifu"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{name} must be an aligned finite numeric or datetime64 vector")
    n_datetime = sum(values.dtype.kind == "M" for values in arrays)
    if n_datetime not in (0, 3):
        raise ValueError("timestamps must all be numeric or all be datetime64")
    if np.any(target[1:] < target[:-1]):
        raise ValueError("CAL target_time must be nondecreasing")
    for first, second in _ORDERED_SPLITS:
        earlier, later = indices[first], indices[second]
        first_issue = issue[later].min()
        if np.any(target[earlier] >= first_issue) or np.any(available[earlier] >= first_issue):
            raise ValueError(
                f"{first} target and availability times must strictly precede "
                f"the earliest {second} issuance"
            )


@dataclass(frozen=True)
class _Schedule:
    rungs: tuple[tuple[int, int], ...]
    policy_updates: int
    custom: tuple[int, ...] | None

    def checkpoints(self, target):
        custom = self.custom
        if custom is not None:
            custom = tuple(update for update in custom if update <= target)
        return _saved_updates(target, custom)


def _schedule(rungs, policy_updates, checkpoint_updates):
    previous_updates, previous_survivors = 0, len(_REGULARIZATION_GRID)
    checked = []
    for updates, survivors in rungs:
        updates = nonnegative_integer(updates, "rung updates")
        survivors = nonnegative_integer(survivors, "rung survivors")
        if updates <= previous_updates or not 1 <= survivors <= previous_survivors:
            raise ValueError("rung targets must increase and survivor counts must be positive and nonincreasing")
        checked.append((updates, survivors))
        previous_updates, previous_survivors = updates, survivors
    if not checked:
        raise ValueError("at least one regularization-search rung is required")
    policy_updates = previous_updates if policy_updates is None else nonnegative_integer(
        policy_updates, "policy_updates"
    )
    if not policy_updates:
        raise ValueError("policy_updates must be positive")
    custom = None
    if checkpoint_updates is not None:
        custom = tuple(sorted({
            nonnegative_integer(update, "checkpoint update") for update in checkpoint_updates
        }))
        if any(update > max(previous_updates, policy_updates) for update in custom):
            raise ValueError("checkpoint updates must not exceed the longest development budget")
    return _Schedule(tuple(checked), policy_updates, custom)


@dataclass(frozen=True)
class _Data:
    baseline: np.ndarray
    lead: np.ndarray
    observed: np.ndarray
    daylight: np.ndarray

    def design(self, rows, geom, state=None):
        return _Design(self.baseline[rows], self.lead[rows], geom, state)

    def score(self, rows, forecasts):
        return score(forecasts, self.observed[rows], daylight=self.daylight[rows])

    def proposal(self, fitting_rows):
        return propose_layouts(fit_pilot(
            self.baseline[fitting_rows], self.observed[fitting_rows], self.lead[fitting_rows]
        ))


@dataclass(frozen=True)
class _Checkpoint:
    updates: int
    qs: float
    theta: np.ndarray
    bridge: np.ndarray

    def forecast(self, design):
        return _issue(design, self.theta, self.bridge)


def _issue(design, theta, bridge):
    raw = design.predict(theta, bridge)
    if not np.isfinite(raw).all():
        raise FloatingPointError("nonfinite development predictions")
    return bounded_pava(raw, merge_means=True)


class _Trial:
    """One uninterrupted solver trajectory with an independent best checkpoint."""

    def __init__(self, train, observed, weights, validation, validation_observed, regularization):
        self.solver = _ChambollePock(train, observed, weights, regularization)
        self.validation = validation
        self.validation_observed = validation_observed
        self.regularization = regularization
        self.updates = self.stale = 0
        self.best = None

    def advance(self, checkpoints):
        for update in checkpoints:
            if update <= self.updates:
                continue
            if self.stale >= _PATIENCE:
                break
            self.solver.advance(update - self.updates)
            self.updates = update
            theta, bridge = self.solver.theta, self.solver.bridge
            improved = False
            if np.any(theta) or np.any(bridge):
                qs = score(_issue(self.validation, theta, bridge), self.validation_observed)["QS"]
                if not np.isfinite(qs):
                    raise FloatingPointError("nonfinite development QS")
                improved = self.best is None or qs < self.best.qs
                if improved:
                    self.best = _Checkpoint(update, qs, theta.copy(), bridge.copy())
            if improved:
                self.stale = 0
            elif update > 4096 and update % 1024 == 0:
                self.stale += 1

    def summary(self):
        return {
            "regularization": self.regularization,
            "updates": self.updates,
            "best_updates": None if self.best is None else self.best.updates,
            "best_qs": None if self.best is None else self.best.qs,
            "patience_stop": self.stale >= _PATIENCE,
        }


def _daylight_rows(data, rows):
    return rows[data.daylight[rows]]


def _search(data, splits, geom, schedule):
    train = data.design(splits.train, geom)
    observed = data.observed[splits.train]
    weights = horizon_weights(data.lead[splits.train])
    stop_rows = _daylight_rows(data, splits.stop)
    stop = data.design(stop_rows, geom, train.state)
    stop_observed = data.observed[stop_rows]
    candidates = (
        _Trial(train, observed, weights, stop, stop_observed, lam)
        for lam in _REGULARIZATION_GRID
    )
    rung_results = []
    for target, limit in schedule.rungs:
        survivors, tried = [], []
        checkpoints = schedule.checkpoints(target)
        for candidate in candidates:
            candidate.advance(checkpoints)
            tried.append(candidate.summary())
            if candidate.best is not None:
                survivors.append(candidate)
                survivors.sort(key=lambda trial: (trial.best.qs, trial.regularization))
                if len(survivors) > limit:
                    survivors.pop()
        if not survivors:
            raise ValueError(f"no nonzero positive-update STOP checkpoint for anchors {geom.anchors}")
        candidates = survivors
        rung_results.append({
            "target_updates": target,
            "trials": tried,
            "promoted_regularization": [trial.regularization for trial in survivors],
        })

    select = data.design(splits.select, geom, train.state)
    eligible = [
        (trial, data.score(splits.select, trial.best.forecast(select))) for trial in candidates
    ]
    chosen, scores = min(eligible, key=lambda pair: (pair[1]["QS"], pair[0].regularization))
    return {
        "anchors": geom.anchors,
        "regularization": chosen.regularization,
        "best_stop_updates": chosen.best.updates,
        "best_stop_qs": chosen.best.qs,
        "select_scores": scores,
        "rungs": rung_results,
        "select_finalists": [
            {**trial.summary(), "select_qs": metrics["QS"]} for trial, metrics in eligible
        ],
    }


def _policy_window(data, fit_rows, dev_rows, regularization, schedule, make_geometry):
    proposal = data.proposal(fit_rows)
    daylight_rows = _daylight_rows(data, dev_rows)
    observed = data.observed[fit_rows]
    weights = horizon_weights(data.lead[fit_rows])
    dev_observed = data.observed[daylight_rows]
    checkpoints = schedule.checkpoints(schedule.policy_updates)
    summary, saved = {}, {}
    for policy, anchors in policy_layouts(proposal).items():
        geom = make_geometry(anchors)
        train = data.design(fit_rows, geom)
        dev = data.design(daylight_rows, geom, train.state)
        trial = _Trial(train, observed, weights, dev, dev_observed, regularization)
        trial.advance(checkpoints)
        if trial.best is None:
            raise ValueError(f"no nonzero positive-update DEV checkpoint for {policy} and anchors {anchors}")
        summary[policy] = {"anchors": anchors, **trial.summary()}
        saved[policy] = (geom, train.state, trial.best)
    return summary, saved


def _dd5(data, splits, schedule, make_geometry):
    train_anchors = data.proposal(splits.train).anchors
    search = _search(data, splits, make_geometry(train_anchors), schedule)
    regularization = search["regularization"]
    warm, _ = _policy_window(
        data, splits.warm_fit, splits.warm_dev, regularization, schedule, make_geometry
    )
    cold, cold_fits = _policy_window(
        data, splits.cold_fit, splits.select, regularization, schedule, make_geometry
    )
    policy = select_policy(
        {name: warm[name]["best_qs"] for name in POLICIES},
        {name: cold[name]["best_qs"] for name in POLICIES},
        {name: warm[name]["best_updates"] for name in POLICIES},
        {name: cold[name]["best_updates"] for name in POLICIES},
    )
    geom, state, checkpoint = cold_fits[policy.policy]
    select = data.design(splits.select, geom, state)
    return {
        "search": search,
        "warm": warm,
        "cold": cold,
        "policy": policy.policy,
        "policy_scores": policy.scores,
        "regularization": regularization,
        "target_updates": policy.target_updates,
        "select_scores": data.score(splits.select, checkpoint.forecast(select)),
        "select_source": "selected policy's cold FIT checkpoint chosen on cold DEV (SELECT)",
    }


def _comparator_scores(comparator_forecasts, data, select_rows):
    if not isinstance(comparator_forecasts, Mapping) or not comparator_forecasts:
        raise ValueError("supply at least one comparator forecast on the common SELECT rows")
    result = {}
    for name, values in comparator_forecasts.items():
        if not isinstance(name, str) or not name.strip() or name in _LAYOUTS:
            raise ValueError("comparator names must be nonempty and distinct from layout names")
        forecast = quantile_matrix(values)
        if len(forecast) != len(select_rows):
            raise ValueError(f"comparator {name} must cover every SELECT row in split order")
        if np.any(forecast[:, 1:] < forecast[:, :-1]) or np.any((forecast < 0) | (forecast > 1)):
            raise ValueError(f"comparator {name} must have bounded and ordered issued quantiles")
        result[name] = data.score(select_rows, forecast)
    return result


def fit_cal(
    quantiles,
    lead_time_min,
    observed,
    daylight,
    splits: CALSplits,
    comparator_forecasts: Mapping[str, ArrayLike],
    *,
    rungs=_DEFAULT_RUNGS,
    policy_updates=None,
    checkpoint_updates=None,
    time_knots=8,
    lead_bounds=_DEFAULT_LEAD_BOUNDS,
    epsilon=_EPSILON,
    target_time=None,
    available_time=None,
    issue_time=None,
) -> tuple[ABQCR, dict]:
    """Develop three layouts within CAL and refit the winner on all CAL rows.

    Provide all-day CAL predictions and observations in per-unit capacity,
    aligned lead minutes, a Boolean daylight mask, and CAL-only index splits.
    Each comparator contains an issued q01–q99 curve for every SELECT index
    in exactly the supplied order. If all three aligned timestamp vectors
    are supplied, check the issuance and observation-availability boundaries.
    Otherwise the caller prepares legal splits before invoking this function.

    The portable CPU schedule uses absolute update budgets. A rung specifies
    (cumulative updates, survivors). Policy trajectories use the final rung
    budget unless policy_updates is supplied. checkpoint_updates replaces the
    interior checkpoint grid, retaining each rung and fit target. STOP and
    DEV patience is evaluated every 1024 updates after update 4096.

    The chosen DD5 policy's cold checkpoint supplies its development SELECT
    forecasts. Select the layout with the lowest mean rank across QS, CRPS,
    CE, and MAE against the common SELECT comparators. Lower QS breaks a
    mean-rank tie, followed by the declared order anchor3, fixed5, dd5.
    The report records all ties and the selected settings. The final model
    saves the full-CAL checkpoint with minimum prepublication training
    objective, including initialization.
    """
    quantiles = quantile_matrix(quantiles)
    n_records = len(quantiles)
    lead = lead_vector(lead_time_min, n_records)
    observed = float_array(observed, "observed")
    daylight = np.asarray(daylight)
    if observed.shape != (n_records,):
        raise ValueError("observed must contain one value per CAL row")
    if daylight.shape != (n_records,) or daylight.dtype.kind != "b":
        raise ValueError("daylight must be a Boolean vector with one value per CAL row")
    timestamps = (target_time, available_time, issue_time)
    splits = _validate_splits(splits, n_records, daylight, timestamps)
    schedule = _schedule(rungs, policy_updates, checkpoint_updates)
    data = _Data(project_quantiles(quantiles), lead, observed, daylight)
    comparators = _comparator_scores(comparator_forecasts, data, splits.select)

    def make_geometry(anchors):
        return geometry(anchors, time_knots, lead_bounds, epsilon)

    developed = {}
    for name, anchors in zip(_LAYOUTS[:2], (THREE_ANCHORS, FIXED_FIVE_ANCHORS)):
        search = _search(data, splits, make_geometry(anchors), schedule)
        developed[name] = {
            "search": search,
            "regularization": search["regularization"],
            "target_updates": search["best_stop_updates"],
            "select_scores": search["select_scores"],
            "select_source": "selected TRAIN fit's STOP checkpoint",
        }
    developed["dd5"] = _dd5(data, splits, schedule, make_geometry)
    layout = select_layout({name: result["select_scores"] for name, result in developed.items()}, comparators)
    name = layout.layout or layout.ties[0]
    winner = developed[name]
    full_cal_proposal = None
    if name == "dd5" and winner["policy"] != "P00":
        full_cal_proposal = data.proposal(slice(None))
        final_anchors = policy_layouts(full_cal_proposal)[winner["policy"]]
    elif name == "dd5":
        final_anchors = FIXED_FIVE_ANCHORS
    else:
        final_anchors = winner["search"]["anchors"]
    target = winner["target_updates"]
    model = ABQCR.fit(
        quantiles, lead, observed, anchors=final_anchors, time_knots=time_knots,
        regularization=winner["regularization"], updates=target, lead_bounds=lead_bounds,
        epsilon=epsilon, checkpoint_updates=schedule.checkpoints(target),
    )
    report = {
        "selected_layout": name,
        "ties": layout.ties,
        "layout_ranks": layout.ranks,
        "layout_mean_ranks": layout.mean_ranks,
        "rank_denominator": layout.denominator,
        "comparator_scores": comparators,
        "development": developed,
        "schedule": {
            "rungs": schedule.rungs,
            "policy_updates": schedule.policy_updates,
            "checkpoint_updates": schedule.custom,
        },
        "calendar_boundaries_checked": target_time is not None,
        "split_records": {name: len(getattr(splits, name)) for name in _SPLIT_FIELDS},
        "full_cal": {
            "records": n_records,
            "anchors": model.anchors,
            "pilot_anchors": None if full_cal_proposal is None else full_cal_proposal.anchors,
            "regularization": model.regularization,
            "target_updates": target,
            "saved_updates": model.fit_summary["saved_updates"],
            "saved_J": model.fit_summary["saved_J"],
        },
    }
    return model, report
