"""Streaming peaks-over-threshold calibration (SPOT, Siffer et al., 2017).

The returned decision cutoff is the mean of the adaptive cutoffs recorded
after each test observation. Detected alarms do not update the tail fit.
"""
from math import log

import numpy as np
from scipy.optimize import minimize


def _candidate_roots(excess, lower, upper):
    """Fit the stationary equation from a regular grid of starting values."""
    if not lower < upper:
        return ()
    step = (upper - lower) / 11
    initial = np.arange(lower + step, upper, step)

    def objective(points):
        cost = 0.0
        gradient = np.zeros_like(points)
        for i, t in enumerate(points):
            values = 1 + t * excess
            u = 1 + np.log(values).mean()
            v = np.mean(1 / values)
            residual = u * v - 1
            du = (1 / t) * (1 - v)
            dv = (1 / t) * (-v + np.mean(1 / values ** 2))
            gradient[i] = 2 * residual * (u * dv + v * du)
            cost += residual ** 2
        return cost, gradient

    solution = minimize(objective, initial, method="L-BFGS-B", jac=True,
                        bounds=[(lower, upper)] * len(initial))
    return np.unique(solution.x)


def _pareto_fit(excess):
    """Choose the tail parameters with the best legacy Grimshaw objective."""
    smallest, largest, average = excess.min(), excess.max(), excess.mean()
    lower = -1 / largest
    epsilon = abs(lower) / 10 if abs(lower) < 2e-8 else 1e-8
    positive_start = 2 * (average - smallest) / (average * smallest)
    positive_end = 2 * (average - smallest) / (smallest ** 2)

    # Preserve the exponential candidate's likelihood convention from training.
    best_shape, best_scale = 0.0, average
    best_likelihood = len(excess) * (1 + log(average))
    for domain in ((lower + epsilon + epsilon, -epsilon), (positive_start, positive_end)):
        for root in _candidate_roots(excess, *domain):
            if root == 0:
                continue
            shape = (1 + np.log(1 + root * excess).mean()) - 1
            scale = shape / root
            if shape == 0 or scale <= 0:
                continue
            likelihood = (-len(excess) * log(scale)
                          - (1 + 1 / shape) * np.log(1 + (shape / scale) * excess).sum())
            if likelihood > best_likelihood:
                best_shape, best_scale, best_likelihood = shape, scale, likelihood
    return best_shape, best_scale


def _cutoff(base, excess, count, risk):
    shape, scale = _pareto_fit(excess)
    tail_ratio = count * risk / len(excess)
    if shape == 0:
        return base - scale * log(tail_ratio)
    return base + (scale / shape) * (pow(tail_ratio, -shape) - 1)


def stream_cutoff(calibration, observations, risk):
    """Return one SPOT decision threshold from two ordered score vectors."""
    if not 0 < risk < 1:
        raise ValueError("risk must be between zero and one")
    baseline = np.asarray(calibration).reshape(-1)
    scores = np.asarray(observations).reshape(-1)
    if not baseline.size or not scores.size:
        raise ValueError("calibration and test scores must be nonempty")
    if not (np.isfinite(baseline).all() and np.isfinite(scores).all()):
        raise ValueError("calibration and test scores must be finite")
    count = len(baseline)
    base = np.sort(baseline)[int(0.98 * count)]
    excess = baseline[baseline > base] - base
    if not excess.size:
        raise ValueError("SPOT requires positive tail excesses above the calibration 98% order statistic")
    cutoff = _cutoff(base, excess, count, risk)
    history = []
    for score in scores:
        if score <= cutoff:
            count += 1
            if score > base:
                excess = np.append(excess, score - base)
                cutoff = _cutoff(base, excess, count, risk)
        history.append(cutoff)
    return float(np.mean(history))
