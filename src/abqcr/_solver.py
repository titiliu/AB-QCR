"""Single-candidate Chambolle–Pock updates for normalized AB-QCR."""

from __future__ import annotations

import math

import numpy as np

from ._geometry import N_QUANTILES, TAU, nonnegative_integer


def horizon_weights(lead):
    _, inverse, counts = np.unique(lead, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def _maximum_gram_eigenvalue(matrix):
    gram = matrix.T @ matrix
    return max(0., float(np.linalg.eigvalsh((gram + gram.T) / 2)[-1]))


def safe_step_bound(design, sqrt_weight):
    geom = design.geom
    blocks = [
        _maximum_gram_eigenvalue(sqrt_weight * design.chi) * _maximum_gram_eigenvalue(geom.P)
    ]
    for b, spline in enumerate(geom.splines):
        block_features = (sqrt_weight * design.amplitude[:, None]) * design.phi[b]
        blocks.append(_maximum_gram_eigenvalue(block_features) * _maximum_gram_eigenvalue(spline))
    upper = sum(blocks)
    if not math.isfinite(upper) or upper <= 0:
        raise FloatingPointError("the weighted operator bound must be positive and finite")
    step = .9 / math.sqrt(upper)
    return upper, step, blocks


def _eigh_psd(gram):
    spectrum, basis = np.linalg.eigh((gram + gram.T) / 2)
    return np.maximum(spectrum, 0.), basis


class _QuadraticProx:
    def __init__(self, geom, step, regularization):
        self.shape = geom.theta_shape
        self.time_knots = len(geom.nodes)
        self.active = any(regularization)
        if self.active:
            lambda_h, lambda_s, lambda_q = regularization
            dh, self.lead_basis = _eigh_psd(geom.lead_penalty)
            ds, self.state_basis = _eigh_psd(geom.state_penalty)
            dq, self.curvature_basis = _eigh_psd(geom.curvature)
            self.main_denom = 1 + (2 * step * lambda_h / geom.main_denominator) * dh[:, None]
            self.bridge_denom = (
                1 + (2 * step * lambda_s / geom.bridge_denominator) * ds[:, None]
                + (2 * step * lambda_q / geom.bridge_denominator) * dq[None, :]
            )

    def __call__(self, theta, bridge):
        if not self.active:
            return theta, bridge
        rotated_main = self.lead_basis.T @ theta.reshape(self.time_knots, -1)
        main = self.lead_basis @ (rotated_main / self.main_denom)
        rotated_detail = self.state_basis.T @ bridge @ self.curvature_basis
        detail = self.state_basis @ (rotated_detail / self.bridge_denom) @ self.curvature_basis.T
        return main.reshape(self.shape), detail


def _regularizers(theta, bridge, geom):
    h = np.diff(theta.reshape(len(geom.nodes), -1), n=2, axis=0)
    return (
        float(np.sum(h ** 2) / geom.main_denominator),
        float(np.sum(bridge * (geom.state_penalty @ bridge)) / geom.bridge_denominator),
        float(np.sum(bridge * (bridge @ geom.curvature)) / geom.bridge_denominator),
    )


def _saved_updates(target, custom):
    if custom is None:
        updates = set(range(256, min(target, 4096) + 1, 256))
        updates.update(range(5120, target + 1, 1024))
    else:
        updates = {nonnegative_integer(value, "checkpoint update") for value in custom}
        if any(value > target for value in updates):
            raise ValueError("checkpoint updates must not exceed the update target")
    return sorted({0, target} | updates)


def _combined_norm(theta, bridge):
    return math.sqrt(float(np.sum(theta ** 2) + np.sum(bridge ** 2)))


class _ChambollePock:
    def __init__(self, design, observed, weights, regularization):
        self.design, self.observed, self.weights = design, observed, weights
        self.regularization = regularization
        geom = design.geom
        self.sqrt_weight = np.sqrt(weights / N_QUANTILES)[:, None]
        self.bound, self.step, self.block_bounds = safe_step_bound(design, self.sqrt_weight)
        self.offset = self.sqrt_weight * (design.baseline - observed[:, None])
        self.dual_lower = -self.sqrt_weight * TAU
        self.dual_upper = self.sqrt_weight * (1 - TAU)
        self.prox = _QuadraticProx(geom, self.step, regularization)
        self.theta = self.bar_theta = np.zeros(geom.theta_shape)
        self.bridge = self.bar_bridge = np.zeros(geom.bridge_shape)
        self.dual = np.zeros_like(design.baseline)

    def _dual_step(self, theta, bridge):
        correction = self.design.forward(theta, bridge)
        proposal = self.dual + self.step * (self.sqrt_weight * correction + self.offset)
        return np.clip(proposal, self.dual_lower, self.dual_upper)

    def _primal_step(self, dual):
        main_grad, bridge_grad = self.design.adjoint(self.sqrt_weight * dual)
        return self.prox(self.theta - self.step * main_grad, self.bridge - self.step * bridge_grad)

    def advance(self, updates):
        for _ in range(updates):
            self.dual = self._dual_step(self.bar_theta, self.bar_bridge)
            theta, bridge = self._primal_step(self.dual)
            self.bar_theta, self.bar_bridge = 2 * theta - self.theta, 2 * bridge - self.bridge
            self.theta, self.bridge = theta, bridge

    def record(self, updates):
        if not all(np.isfinite(block).all() for block in (
            self.theta, self.bridge, self.bar_theta, self.bar_bridge, self.dual
        )):
            raise FloatingPointError("nonfinite Chambolle–Pock state")
        error = self.observed[:, None] - self.design.predict(self.theta, self.bridge)
        pinball = np.maximum(TAU * error, (TAU - 1) * error)
        loss = float(np.sum(self.weights[:, None] * pinball) / N_QUANTILES)
        rh, rs, rq = _regularizers(self.theta, self.bridge, self.design.geom)
        objective = loss + sum(lam * reg for lam, reg in zip(self.regularization, (rh, rs, rq)))

        # Current-point diagnostics leave both primal and dual states untouched.
        theta, bridge = self._primal_step(self.dual)
        p = _combined_norm(theta - self.theta, bridge - self.bridge) / self.step
        q = float(np.linalg.norm(self._dual_step(self.theta, self.bridge) - self.dual)) / self.step
        row = {
            "updates": updates, "L": loss, "R_h": rh, "R_s": rs, "R_q": rq, "J": objective,
            "primal_fixed_point": p, "dual_fixed_point": q,
            "relative_primal": p / (1 + _combined_norm(self.theta, self.bridge)),
            "relative_dual": q / (1 + float(np.linalg.norm(self.dual))),
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("nonfinite objective or fixed-point diagnostic")
        return row


def solve(design, observed, weights, regularization, updates, checkpoint_updates, selection):
    checkpoints = _saved_updates(updates, checkpoint_updates)
    cp = _ChambollePock(design, observed, weights, regularization)
    trace = []
    previous = 0
    best_point = None
    for update in checkpoints:
        cp.advance(update - previous)
        point = cp.record(update)
        trace.append(point)
        if best_point is None or point["J"] < best_point["J"]:
            best_point, best_theta, best_bridge = point, cp.theta, cp.bridge
        previous = update
    if selection == "last":
        saved, theta, bridge = trace[-1], cp.theta, cp.bridge
    else:
        saved, theta, bridge = best_point, best_theta, best_bridge
    summary = {
        "selection": selection,
        "target_updates": updates,
        "actual_updates": updates,
        "saved_updates": saved["updates"],
        "initial_J": trace[0]["J"],
        "final_J": trace[-1]["J"],
        "saved_J": saved["J"],
        "L2_upper": cp.bound,
        "step": cp.step,
        "sigma": cp.step,
        "block_bounds": cp.block_bounds,
        "snapshots": trace,
    }
    return theta, bridge, summary
