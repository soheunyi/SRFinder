"""Class-specific multiplier KS test with continuous affine-nuisance inversion.

This tests the CONDITIONAL weighted-distribution null
    F4 = t Fplus + (1-t) Fminus for some t in [0, 1].
It does not establish that a physical no-signal null implies this model.

Assumptions: independent event records within and between the two SR samples;
nonnegative weights; positive endpoint-weight totals; sufficient weight moments
and no dominant event for the empirical-process approximation. Scores, SR,
and learned transfer/calibration functions are conditioned on. Repeated rows
from a common parent event are NOT independent records: this implementation
must not be used unchanged for that design.

The bootstrap is first-order/asymptotic, not finite-sample exact. The search
in t compares piecewise-linear envelopes, not an arbitrary parameter grid.
A tiny outward numerical tolerance makes floating-point comparisons conservative;
it is not a bias allowance for background-transfer error.

Requires Python >= 3.10 and NumPy. No SciPy, neural-network retraining, or bins.
The per-replicate envelope construction takes O(m log m) time, with m distinct
pooled scores; it can be slow for very large datasets. Replicates are streamed.

General references (the particular test is derived in the accompanying answer):
Praestgaard and Wellner (1993), Annals of Probability 21, 2053-2086.
Berger and Boos (1994), JASA 89, 1012-1016 (nuisance maximization principle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

LD = np.longdouble


@dataclass(frozen=True)
class AffineKSResult:
    p_value: float
    reject: bool
    alpha: float
    ks_statistic: float
    ks_t: float
    maximizing_p_t: float
    max_exceedances: int
    bootstrap_replicates: int
    n3: int
    n4: int
    distinct_scores: int
    effective_sample_sizes: dict[str, float]
    max_normalized_weights: dict[str, float]
    numerical_tolerance: float


@dataclass(frozen=True)
class _Envelope:
    left: NDArray[Any]
    right: NDArray[Any]
    slope: NDArray[Any]
    intercept: NDArray[Any]


def _upper_envelope(slopes: ArrayLike, intercepts: ArrayLike) -> _Envelope:
    """Upper envelope of finitely many lines restricted to [0, 1]."""
    slopes = np.asarray(slopes, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    order = np.argsort(slopes, kind="stable")
    s = slopes[order]
    a = intercepts[order]
    first = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    a = np.maximum.reduceat(a, first)
    s = s[first]
    hs = np.empty(s.size, dtype=LD)
    ha = np.empty(s.size, dtype=LD)
    hx = np.empty(s.size, dtype=LD)
    k = 0
    for si, ai in zip(s, a):
        si, ai = LD(si), LD(ai)
        x = LD(-np.inf)
        while k:
            x = (ha[k - 1] - ai) / (si - hs[k - 1])
            if x > hx[k - 1]:
                break
            k -= 1
        if not k:
            x = LD(-np.inf)
        hs[k], ha[k], hx[k] = si, ai, x
        k += 1
    hs, ha, hx = hs[:k], ha[:k], hx[:k]
    start = int(np.searchsorted(hx, LD(0), side="right") - 1)
    end = int(np.searchsorted(hx, LD(1), side="left"))
    ids = np.arange(start, end)
    left = np.maximum(hx[ids], LD(0))
    next_x = np.r_[hx[1:], LD(np.inf)]
    right = np.minimum(next_x[ids], LD(1))
    return _Envelope(left, right, hs[ids], ha[ids])


def _norm_envelope(a: NDArray[Any], d: NDArray[Any]) -> _Envelope:
    return _upper_envelope(np.r_[d, -d], np.r_[a, -a])


def _envelope_minimum(e: _Envelope) -> tuple[float, float]:
    x = np.r_[e.left, e.right]
    values = np.r_[e.intercept + e.slope * e.left,
                   e.intercept + e.slope * e.right]
    j = int(np.argmin(values))
    return max(0.0, float(values[j])), float(x[j])


def _exceedance_intervals(
    observed: _Envelope, bootstrap: _Envelope, tol: float
) -> list[tuple[np.longdouble, np.longdouble]]:
    """Closed interval union where Q_boot(t) + tol >= D_observed(t)."""
    intervals: list[tuple[np.longdouble, np.longdouble]] = []
    i = j = 0
    while i < observed.left.size and j < bootstrap.left.size:
        lo = max(observed.left[i], bootstrap.left[j])
        hi = min(observed.right[i], bootstrap.right[j])
        d = bootstrap.slope[j] - observed.slope[i]
        a = bootstrap.intercept[j] - observed.intercept[i] + LD(tol)
        if lo <= hi:
            flo, fhi = a + d * lo, a + d * hi
            interval = None
            if flo >= 0 and fhi >= 0:
                interval = (lo, hi)
            elif flo >= 0:
                interval = (lo, min(hi, max(lo, -a / d)))
            elif fhi >= 0:
                interval = (min(hi, max(lo, -a / d)), hi)
            if interval is not None:
                left, right = interval
                # Merge within each replicate, so one draw is never counted twice.
                if intervals and left <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(right, intervals[-1][1]))
                else:
                    intervals.append((left, right))
        old_o, old_b = observed.right[i], bootstrap.right[j]
        if old_o <= old_b:
            i += 1
        if old_b <= old_o:
            j += 1
    return intervals


def _max_overlap(
    intervals: list[tuple[np.longdouble, np.longdouble]],
) -> tuple[int, float]:
    """Count maximum overlap of closed intervals; starts precede tied ends."""
    if not intervals:
        return 0, 0.0
    events = [(lo, 0) for lo, _ in intervals]
    events += [(hi, 1) for _, hi in intervals]
    events.sort()
    current = best = 0
    best_t = LD(0)
    for location, kind in events:
        if kind == 0:
            current += 1
            if current > best:
                best, best_t = current, location
        else:
            current -= 1
    return best, float(best_t)


def _validate_sample(
    scores: ArrayLike, weights: ArrayLike, name: str, L: float, U: float
) -> tuple[NDArray[Any], NDArray[Any]]:
    z = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or w.ndim != 1 or z.size != w.size or z.size == 0:
        raise ValueError(f"{name}: scores and weights must be nonempty 1D arrays of equal size.")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)):
        raise ValueError(f"{name}: all scores and weights must be finite; handle clipping beforehand.")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError(f"{name}: weights must be nonnegative with at least one positive weight.")
    if np.any(z < L) or np.any(z > U):
        raise ValueError(f"{name}: scores must already lie within the specified support [L, U].")
    order = np.argsort(z, kind="stable")
    # Scaling the entire class changes neither the statistic nor the bootstrap.
    return z[order], w[order] / w.max()


def affine_ks_test(
    z3: ArrayLike,
    w3: ArrayLike,
    z4: ArrayLike,
    w4: ArrayLike,
    *,
    L: float,
    U: float,
    B: int = 1999,
    alpha: float = 0.05,
    seed: int | None = None,
    numerical_tol: float = 1e-12,
) -> AffineKSResult:
    """Test the weighted affine-family null using event-level Poisson multipliers.

    w3 is the full saved weight: supplied event weight times learned CR ratio.
    w4 is the saved 4b event weight. Do not apply either weight twice.
    SR membership must have been determined before clipping the saved scores.

    Returns a Monte Carlo p-value maximized over the continuous interval [0, 1],
    using piecewise-linear envelope intersection rather than a parameter grid.
    numerical_tol adds only an outward tolerance to bootstrap comparisons.
    """
    if not np.isfinite(L) or not np.isfinite(U) or not L < U:
        raise ValueError("Require finite L < U.")
    if not np.isfinite(U - L):
        raise ValueError("The support width U-L must be finite.")
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError("B must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol must be finite and nonnegative.")
    z3s, w3s = _validate_sample(z3, w3, "3b", L, U)
    z4s, w4s = _validate_sample(z4, w4, "4b", L, U)
    x3 = (z3s - L) / (U - L)
    plus = w3s * x3
    minus = w3s * (1 - x3)
    if plus.sum() <= 0 or minus.sum() <= 0:
        raise ValueError("Both 3b endpoint-tilted weight totals must be positive.")
    up = plus / plus.sum()
    um = minus / minus.sum()
    r = w4s / w4s.sum()
    y = np.unique(np.r_[z3s, z4s])
    i3 = np.searchsorted(z3s, y, side="right")
    i4 = np.searchsorted(z4s, y, side="right")

    def cdf(q: NDArray[Any], idx: NDArray[Any]) -> NDArray[Any]:
        cumulative = np.r_[0.0, np.cumsum(q, dtype=np.float64)]
        cumulative[-1] = 1.0
        return cumulative[idx]

    fp, fm, f4 = cdf(up, i3), cdf(um, i3), cdf(r, i4)
    observed = _norm_envelope(fm - f4, fp - fm)
    ks, ks_t = _envelope_minimum(observed)
    rng = np.random.default_rng(seed)
    all_intervals: list[tuple[np.longdouble, np.longdouble]] = []

    def process(xi: NDArray[Any], q: NDArray[Any], f: NDArray[Any],
                idx: NDArray[Any]) -> NDArray[Any]:
        cumulative = np.r_[0.0, np.cumsum(xi * q, dtype=np.float64)]
        return cumulative[idx] - f * cumulative[-1]

    for _ in range(B):
        xi3 = rng.poisson(1.0, size=z3s.size) - 1
        xi4 = rng.poisson(1.0, size=z4s.size) - 1
        # Multipliers are assigned to original events BEFORE tied scores are grouped.
        gp = process(xi3, up, fp, i3)
        gm = process(xi3, um, fm, i3)
        g4 = process(xi4, r, f4, i4)
        boot = _norm_envelope(gm - g4, gp - gm)
        all_intervals.extend(_exceedance_intervals(observed, boot, numerical_tol))

    count, p_t = _max_overlap(all_intervals)
    p_value = (1 + count) / (B + 1)
    q_by_name = {"3b_plus": up, "3b_minus": um, "4b": r}
    return AffineKSResult(
        p_value=p_value,
        reject=bool(p_value <= alpha),
        alpha=float(alpha),
        ks_statistic=ks,
        ks_t=ks_t,
        maximizing_p_t=p_t,
        max_exceedances=count,
        bootstrap_replicates=int(B),
        n3=int(z3s.size),
        n4=int(z4s.size),
        distinct_scores=int(y.size),
        effective_sample_sizes={name: float(1 / np.dot(q, q))
                                for name, q in q_by_name.items()},
        max_normalized_weights={name: float(q.max()) for name, q in q_by_name.items()},
        numerical_tolerance=float(numerical_tol),
    )
