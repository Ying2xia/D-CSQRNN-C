"""Data generation and censoring utilities for Chapter 5."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .config import CENSORING_SPECS, CensoringSpec, LevelDist


NORMAL_ERROR_SIGMA = 1.0


@dataclass
class SimDataset:
    X: np.ndarray
    y_true: np.ndarray
    y_obs: np.ndarray
    delta: np.ndarray
    L: np.ndarray
    R: np.ndarray
    scenario: int
    error: str
    censor_type: str | None = None
    censor_rate: float | None = None

    @property
    def uncensored_mask(self) -> np.ndarray:
        return self.delta == 0

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def generate_xy(n: int, scenario: int, error: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate fully observed (X, y) from the two mechanisms in Section 5.1."""

    if error == "normal":
        eps = rng.normal(0.0, NORMAL_ERROR_SIGMA, size=n)
    elif error == "t3":
        eps = rng.standard_t(df=3, size=n)
    else:
        raise ValueError(f"unknown error distribution: {error}")

    if scenario == 1:
        X = rng.normal(0.0, 0.5, size=(n, 2))
        x1, x2 = X[:, 0], X[:, 1]
        y = np.sin(2.0 * x1) + 2.0 * np.exp(-16.0 * x2**2) + 0.5 * eps
    elif scenario == 2:
        X = rng.uniform(-1.0, 1.0, size=(n, 2))
        x1, x2 = X[:, 0], X[:, 1]
        mean = (1.0 - x1 + 2.0 * x2**2) * np.exp(-0.5 * (x1 + x2) ** 2)
        scale = (1.0 + 0.2 * (x1 + x2)) / 5.0
        y = mean + scale * eps
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return X.astype(float), y.astype(float)


def _scenario_location_scale(X: np.ndarray, scenario: int) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional location and error multiplier used by ``generate_xy``."""

    if scenario == 1:
        x1, x2 = X[:, 0], X[:, 1]
        loc = np.sin(2.0 * x1) + 2.0 * np.exp(-16.0 * x2**2)
        scale = np.full(X.shape[0], 0.5, dtype=float)
        return loc, scale
    if scenario == 2:
        x1, x2 = X[:, 0], X[:, 1]
        loc = (1.0 - x1 + 2.0 * x2**2) * np.exp(-0.5 * (x1 + x2) ** 2)
        scale = (1.0 + 0.2 * (x1 + x2)) / 5.0
        return loc, scale
    raise ValueError(f"unknown scenario: {scenario}")


def _t3_cdf(x: float) -> float:
    """Closed-form CDF of Student's t distribution with 3 degrees of freedom."""

    root3 = np.sqrt(3.0)
    return float(0.5 + (np.arctan(x / root3) + x * root3 / (x * x + 3.0)) / np.pi)


def _t3_ppf(prob: float) -> float:
    """Numerically invert the closed-form t(3) CDF.

    Only a few fixed tau values are needed in the simulations, so a small
    bisection solver is simpler and more portable than depending on SciPy.
    """

    p = float(prob)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if p == 0.5:
        return 0.0

    lo, hi = (-1.0, 1.0)
    while _t3_cdf(lo) > p:
        lo *= 2.0
    while _t3_cdf(hi) < p:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _t3_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def true_conditional_quantiles(
    X: np.ndarray,
    scenario: int,
    error: str,
    taus: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """True conditional quantiles for the Chapter 5 simulation mechanisms.

    The returned array has shape ``(n_observations, n_taus)`` and is used only
    for simulation diagnostics such as bootstrap-CI coverage. Real data do not
    have access to this quantity.
    """

    loc, scale = _scenario_location_scale(np.asarray(X, dtype=float), int(scenario))
    taus_arr = np.asarray(taus, dtype=float)
    if error == "normal":
        normal = NormalDist(mu=0.0, sigma=NORMAL_ERROR_SIGMA)
        err_q = np.array([normal.inv_cdf(float(tau)) for tau in taus_arr], dtype=float)
    elif error == "t3":
        err_q = np.array([_t3_ppf(float(tau)) for tau in taus_arr], dtype=float)
    else:
        raise ValueError(f"unknown error distribution: {error}")
    return loc[:, None] + scale[:, None] * err_q[None, :]


def as_fully_observed(dataset: SimDataset) -> SimDataset:
    """Return a fully observed copy of a simulated censored dataset."""

    nan = np.full(dataset.n, np.nan)
    return SimDataset(
        X=dataset.X,
        y_true=dataset.y_true,
        y_obs=dataset.y_true.copy(),
        delta=np.zeros(dataset.n, dtype=int),
        L=nan.copy(),
        R=nan.copy(),
        scenario=dataset.scenario,
        error=dataset.error,
        censor_type=None,
        censor_rate=None,
    )


def make_uncensored_dataset(n: int, scenario: int, error: str, rng: np.random.Generator) -> SimDataset:
    X, y = generate_xy(n, scenario, error, rng)
    nan = np.full(n, np.nan)
    return SimDataset(
        X=X,
        y_true=y,
        y_obs=y.copy(),
        delta=np.zeros(n, dtype=int),
        L=nan.copy(),
        R=nan.copy(),
        scenario=scenario,
        error=error,
    )


def _draw_level(dist: LevelDist | None, n: int, rng: np.random.Generator) -> np.ndarray:
    if dist is None:
        return np.full(n, np.nan)
    if dist.name == "normal":
        if dist.b is None:
            raise ValueError("normal censoring levels require a standard deviation")
        return rng.normal(dist.a, dist.b, size=n)
    if dist.name == "exp":
        return rng.exponential(scale=1.0 / dist.a, size=n)
    raise ValueError(f"unknown censoring level distribution: {dist.name}")


def apply_censoring(
    X: np.ndarray,
    y: np.ndarray,
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    rng: np.random.Generator,
    spec: CensoringSpec | None = None,
) -> SimDataset:
    """Apply left, right, or interval censoring using Table 5.1 distributions."""

    if spec is None:
        spec = CENSORING_SPECS[(scenario, error, censor_type, float(censor_rate))]

    n = len(y)
    L = _draw_level(spec.left, n, rng)
    R = _draw_level(spec.right, n, rng)
    delta = np.zeros(n, dtype=int)
    y_obs = y.copy()

    if censor_type == "left":
        censored = y <= L
        delta[censored] = 1
        y_obs[censored] = L[censored]
    elif censor_type == "right":
        censored = y >= R
        delta[censored] = 2
        y_obs[censored] = R[censored]
    elif censor_type == "interval":
        low = np.minimum(L, R)
        high = np.maximum(L, R)
        L, R = low, high
        censored = (y >= L) & (y <= R)
        delta[censored] = 3
        y_obs[censored] = 0.5 * (L[censored] + R[censored])
    else:
        raise ValueError(f"unknown censoring type: {censor_type}")

    return SimDataset(
        X=X,
        y_true=y,
        y_obs=y_obs,
        delta=delta,
        L=L,
        R=R,
        scenario=scenario,
        error=error,
        censor_type=censor_type,
        censor_rate=float(censor_rate),
    )


def make_censored_dataset(
    n: int,
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    rng: np.random.Generator,
) -> SimDataset:
    X, y = generate_xy(n, scenario, error, rng)
    return apply_censoring(X, y, scenario, error, censor_type, censor_rate, rng)


def make_tau_grid(n_train: int) -> np.ndarray:
    k_tau = max(int(np.ceil(np.sqrt(n_train))), 100)
    k = np.arange(1, k_tau + 1, dtype=float)
    return k / (k_tau + 1.0)


def boundary_values(dataset: SimDataset) -> np.ndarray:
    """Boundary fallback values used when a random quantile prediction is infeasible."""

    out = dataset.y_obs.copy()
    left = dataset.delta == 1
    right = dataset.delta == 2
    interval = dataset.delta == 3
    out[left] = dataset.L[left]
    out[right] = dataset.R[right]
    out[interval] = 0.5 * (dataset.L[interval] + dataset.R[interval])
    return out


def feasible(values: np.ndarray, dataset: SimDataset) -> np.ndarray:
    ok = np.ones(dataset.n, dtype=bool)
    left = dataset.delta == 1
    right = dataset.delta == 2
    interval = dataset.delta == 3
    ok[left] = values[left] <= dataset.L[left]
    ok[right] = values[right] >= dataset.R[right]
    ok[interval] = (values[interval] >= dataset.L[interval]) & (values[interval] <= dataset.R[interval])
    return ok


def impute_from_predictions(pred: np.ndarray, dataset: SimDataset) -> np.ndarray:
    """Use one predicted quantile for censored observations, otherwise boundary fallback."""

    y_imp = dataset.y_obs.copy()
    censored = dataset.delta != 0
    ok = feasible(pred, dataset)
    fallback = boundary_values(dataset)
    y_imp[censored & ok] = pred[censored & ok]
    y_imp[censored & ~ok] = fallback[censored & ~ok]
    return y_imp


def impute_from_candidate_predictions(
    predictions: np.ndarray,
    dataset: SimDataset,
    rng: np.random.Generator,
) -> np.ndarray:
    """Impute censored observations from several already-trained quantile models.

    Vectorized version: avoids the per-observation Python loop, which becomes
    a major bottleneck for distributed simulations with n_train = 200,000.
    """

    pred = np.asarray(predictions, dtype=float)
    if pred.ndim == 1:
        return impute_from_predictions(pred, dataset)
    n, n_cand = pred.shape
    if n != dataset.n:
        raise ValueError("candidate predictions must have shape (n_observations, n_candidates)")

    y_imp = dataset.y_obs.copy()
    fallback = boundary_values(dataset)
    delta = dataset.delta
    L = dataset.L
    R = dataset.R

    # Vectorized feasibility check (n, n_cand)
    is_left = (delta == 1)
    is_right = (delta == 2)
    is_int = (delta == 3)
    censored = (delta != 0)

    ok = np.zeros((n, n_cand), dtype=bool)
    if is_left.any():
        ok[is_left] = pred[is_left] <= L[is_left, None]
    if is_right.any():
        ok[is_right] = pred[is_right] >= R[is_right, None]
    if is_int.any():
        ok[is_int] = (pred[is_int] >= L[is_int, None]) & (pred[is_int] <= R[is_int, None])
    ok &= np.isfinite(pred)

    n_feas = ok.sum(axis=1)
    has_feas = censored & (n_feas > 0)

    # For each row with feasible values, pick a random one.
    # Use cumsum + matching to vectorize the "pick the k-th True" operation.
    rand_idx = rng.integers(0, np.maximum(n_feas, 1), size=n)
    cum = np.cumsum(ok, axis=1) - 1  # 0-indexed running count of True values
    matches = (cum == rand_idx[:, None]) & ok
    # argmax picks the first True; for has_feas rows there is exactly one True
    sampled_col = matches.argmax(axis=1)
    rows_with_feas = np.flatnonzero(has_feas)
    y_imp[rows_with_feas] = pred[rows_with_feas, sampled_col[rows_with_feas]]

    # Boundary fallback for censored rows that have no feasible candidate
    no_feas = censored & ~has_feas
    if no_feas.any():
        y_imp[no_feas] = fallback[no_feas]
    return y_imp


def subset_dataset(dataset: SimDataset, idx: np.ndarray) -> SimDataset:
    return SimDataset(
        X=dataset.X[idx],
        y_true=dataset.y_true[idx],
        y_obs=dataset.y_obs[idx],
        delta=dataset.delta[idx],
        L=dataset.L[idx],
        R=dataset.R[idx],
        scenario=dataset.scenario,
        error=dataset.error,
        censor_type=dataset.censor_type,
        censor_rate=dataset.censor_rate,
    )
