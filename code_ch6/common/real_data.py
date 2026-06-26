"""Real-data loading, splitting, standardization, and artificial censoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .data import SimDataset
from .real_config import CENTRAL_CENSORING_SPECS, HOUSEHOLD_CENSORING_SPECS, RealCensoringSpec


RDATSETS_RAW = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
HOUSEHOLD_FILE = "household_power_consumption.txt"


@dataclass
class RealRawDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    response_name: str

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass
class RealSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_index: np.ndarray
    test_index: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float


def _cache_path(cache_dir: Path, package: str, dataset: str) -> Path:
    return cache_dir / f"{package}_{dataset}.csv"


def _read_rdataset(package: str, dataset: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, package, dataset)
    if cache.exists():
        return pd.read_csv(cache)
    url = f"{RDATSETS_RAW}/{package}/{dataset}.csv"
    df = pd.read_csv(url)
    df.to_csv(cache, index=False)
    return df


def _drop_rdataset_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["rownames", "Unnamed: 0", "index"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def load_boston(cache_dir: str | Path) -> RealRawDataset:
    """Load MASS::Boston from the Rdatasets mirror."""

    df = _drop_rdataset_index(_read_rdataset("MASS", "Boston", Path(cache_dir)))
    response = "medv" if "medv" in df.columns else "MEDV"
    if response not in df.columns:
        raise ValueError(f"Boston response column not found; columns={list(df.columns)}")
    X = df.drop(columns=[response]).to_numpy(dtype=float)
    y = df[response].to_numpy(dtype=float)
    return RealRawDataset("boston", X, y, list(df.drop(columns=[response]).columns), response)


def load_gilgais(cache_dir: str | Path) -> RealRawDataset:
    """Load MASS::gilgais from the Rdatasets mirror."""

    df = _drop_rdataset_index(_read_rdataset("MASS", "gilgais", Path(cache_dir)))
    response = "e80"
    if response not in df.columns:
        raise ValueError(f"Gilgais response column 'e80' not found; columns={list(df.columns)}")
    feature_order = ["pH00", "pH30", "pH80", "e00", "e30", "c00", "c30", "c80"]
    features = [col for col in feature_order if col in df.columns]
    if len(features) != len(feature_order):
        features = [col for col in df.columns if col != response]
    X = df[features].to_numpy(dtype=float)
    y = df[response].to_numpy(dtype=float)
    return RealRawDataset("gilgais", X, y, features, response)


def load_household_power(data_dir: str | Path, max_rows: int | None = None) -> RealRawDataset:
    """Load the UCI household power-consumption data already placed in code_ch6."""

    path = Path(data_dir) / HOUSEHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"household power data not found: {path}")
    columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    df = pd.read_csv(
        path,
        sep=";",
        usecols=columns,
        na_values=["?"],
        nrows=max_rows,
        low_memory=False,
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)
    response = "Global_active_power"
    features = [col for col in columns if col != response]
    return RealRawDataset(
        "household_power",
        df[features].to_numpy(dtype=float),
        df[response].to_numpy(dtype=float),
        features,
        response,
    )


def load_real_dataset(name: str, data_dir: str | Path, cache_dir: str | Path | None = None, max_rows: int | None = None) -> RealRawDataset:
    key = name.lower().replace("-", "_")
    root = Path(data_dir)
    cache = Path(cache_dir) if cache_dir is not None else root / "data_cache"
    if key in {"boston", "bostonhousing", "boston_housing"}:
        return load_boston(cache)
    if key in {"gilgais", "gilgai"}:
        return load_gilgais(cache)
    if key in {"household", "household_power", "electricity", "power"}:
        return load_household_power(root, max_rows=max_rows)
    raise ValueError(f"unknown real dataset: {name}")


def split_and_standardize(
    data: RealRawDataset,
    rng: np.random.Generator,
    train_fraction: float = 0.8,
    n_train: int | None = None,
    n_test: int | None = None,
) -> RealSplit:
    """Random split and standardize X and y using training-set statistics."""

    n = data.n
    if n_train is None:
        n_train = int(np.floor(float(train_fraction) * n))
    n_train = min(int(n_train), n - 1)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    remaining = perm[n_train:]
    if n_test is not None:
        remaining = remaining[: min(int(n_test), len(remaining))]
    test_idx = remaining
    if len(test_idx) == 0:
        raise ValueError("test set is empty; reduce n_train or supply more rows")

    X_train_raw = data.X[train_idx]
    X_test_raw = data.X[test_idx]
    y_train_raw = data.y[train_idx]
    y_test_raw = data.y[test_idx]

    x_mean = X_train_raw.mean(axis=0)
    x_scale = X_train_raw.std(axis=0)
    x_scale = np.where(x_scale <= 1e-8, 1.0, x_scale)
    y_mean = float(y_train_raw.mean())
    y_scale = float(y_train_raw.std())
    y_scale = y_scale if y_scale > 1e-8 else 1.0

    return RealSplit(
        X_train=(X_train_raw - x_mean) / x_scale,
        y_train=(y_train_raw - y_mean) / y_scale,
        X_test=(X_test_raw - x_mean) / x_scale,
        y_test=(y_test_raw - y_mean) / y_scale,
        train_index=train_idx,
        test_index=test_idx,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )


def _draw_normal_level(mean: float, sd: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(float(mean), float(sd), size=n)


def _draw_from_spec(spec: RealCensoringSpec, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    L = np.full(n, np.nan)
    R = np.full(n, np.nan)
    if spec.left is not None:
        if spec.left.name != "normal" or spec.left.b is None:
            raise ValueError("Chapter 6 real-data censoring specs currently expect normal levels.")
        L = _draw_normal_level(spec.left.a, spec.left.b, n, rng)
    if spec.right is not None:
        if spec.right.name != "normal" or spec.right.b is None:
            raise ValueError("Chapter 6 real-data censoring specs currently expect normal levels.")
        R = _draw_normal_level(spec.right.a, spec.right.b, n, rng)
    return L, R


def _normal_ppf(prob: float) -> float:
    return NormalDist().inv_cdf(float(prob))


def _calibrated_household_levels(y: np.ndarray, censor_type: str, censor_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic boundaries that hit the nominal censoring rate closely.

    This is kept as a sensitivity helper. The thesis scripts use random
    distribution-generated household censoring levels from
    ``HOUSEHOLD_CENSORING_SPECS``.
    """

    n = len(y)
    rate = float(censor_rate)
    L = np.full(n, np.nan)
    R = np.full(n, np.nan)
    if censor_type == "left":
        L[:] = np.quantile(y, rate)
    elif censor_type == "right":
        R[:] = np.quantile(y, 1.0 - rate)
    elif censor_type == "interval":
        L[:] = np.quantile(y, (1.0 - rate) / 2.0)
        R[:] = np.quantile(y, (1.0 + rate) / 2.0)
    else:
        raise ValueError(f"unknown censoring type: {censor_type}")
    return L, R


def apply_real_censoring(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    censor_type: str,
    censor_rate: float,
    rng: np.random.Generator,
) -> SimDataset:
    """Apply artificial censoring on the standardized response scale."""

    n = len(y)
    key = dataset_name.lower()
    if key in {"boston", "gilgais"}:
        spec = CENTRAL_CENSORING_SPECS[(key, censor_type)]
        L, R = _draw_from_spec(spec, n, rng)
    elif key in {"household_power", "household", "electricity", "power"}:
        spec = HOUSEHOLD_CENSORING_SPECS[(censor_type, float(censor_rate))]
        L, R = _draw_from_spec(spec, n, rng)
    else:
        L, R = _calibrated_household_levels(y, censor_type, censor_rate)

    y_obs = y.copy()
    delta = np.zeros(n, dtype=int)
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
        X=np.asarray(X, dtype=float),
        y_true=np.asarray(y, dtype=float),
        y_obs=y_obs,
        delta=delta,
        L=L,
        R=R,
        scenario=0,
        error=dataset_name,
        censor_type=censor_type,
        censor_rate=float(censor_rate),
    )


def censoring_level_summary(dataset: SimDataset) -> dict[str, float]:
    out = {
        "actual_censor_rate": float((dataset.delta != 0).mean()),
        "L_mean": float(np.nanmean(dataset.L)) if np.isfinite(dataset.L).any() else np.nan,
        "L_sd": float(np.nanstd(dataset.L)) if np.isfinite(dataset.L).any() else np.nan,
        "R_mean": float(np.nanmean(dataset.R)) if np.isfinite(dataset.R).any() else np.nan,
        "R_sd": float(np.nanstd(dataset.R)) if np.isfinite(dataset.R).any() else np.nan,
    }
    if np.isfinite(dataset.L).any():
        out["L_q25"] = float(np.nanquantile(dataset.L, 0.25))
        out["L_q50"] = float(np.nanquantile(dataset.L, 0.50))
        out["L_q75"] = float(np.nanquantile(dataset.L, 0.75))
    if np.isfinite(dataset.R).any():
        out["R_q25"] = float(np.nanquantile(dataset.R, 0.25))
        out["R_q50"] = float(np.nanquantile(dataset.R, 0.50))
        out["R_q75"] = float(np.nanquantile(dataset.R, 0.75))
    return out
