"""Evaluation metrics and raw-result summarisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import check_loss_values
from .training import tau_key


def quantile_loss(y_true: np.ndarray, pred: np.ndarray, tau: float) -> float:
    return float(check_loss_values(y_true - pred, tau).mean())


def interval_bounds(iter_predictions: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Prediction interval bounds from raw iteration/bootstrap predictions."""

    lower = np.quantile(iter_predictions, alpha / 2.0, axis=0)
    upper = np.quantile(iter_predictions, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def piw(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.mean(upper - lower, axis=0)


def interval_coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of fixed targets by pointwise interval bounds."""

    target = np.asarray(target, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ok = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
    if not ok.any():
        return float("nan")
    return float(((target[ok] >= lower[ok]) & (target[ok] <= upper[ok])).mean())


def metric_rows_from_predictions(
    y_test: np.ndarray,
    target_taus: list[float],
    pred_censored: np.ndarray,
    pred_benchmark: np.ndarray,
    lower_censored: np.ndarray,
    upper_censored: np.ndarray,
    lower_benchmark: np.ndarray,
    upper_benchmark: np.ndarray,
    metadata: dict[str, object],
    true_quantiles: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    piw_c = piw(lower_censored, upper_censored)
    piw_u = piw(lower_benchmark, upper_benchmark)
    true_q = None if true_quantiles is None else np.asarray(true_quantiles, dtype=float)
    for j, tau in enumerate(target_taus):
        ql_c = quantile_loss(y_test, pred_censored[:, j], tau)
        ql_u = quantile_loss(y_test, pred_benchmark[:, j], tau)
        coverage = float("nan")
        benchmark_coverage = float("nan")
        if true_q is not None:
            coverage = interval_coverage(true_q[:, j], lower_censored[:, j], upper_censored[:, j])
            benchmark_coverage = interval_coverage(true_q[:, j], lower_benchmark[:, j], upper_benchmark[:, j])
        row = dict(metadata)
        row.update(
            {
                "tau": tau_key(tau),
                "ql_censored": ql_c,
                "ql_benchmark": ql_u,
                "ql_ratio": ql_c / ql_u if ql_u > 0 else np.nan,
                "lower_censored_mean": float(lower_censored[:, j].mean()),
                "upper_censored_mean": float(upper_censored[:, j].mean()),
                "lower_benchmark_mean": float(lower_benchmark[:, j].mean()),
                "upper_benchmark_mean": float(upper_benchmark[:, j].mean()),
                "piw_censored": float(piw_c[j]),
                "piw_benchmark": float(piw_u[j]),
                "piw_ratio": float(piw_c[j] / piw_u[j]) if piw_u[j] > 0 else np.nan,
                "coverage": coverage,
                "benchmark_coverage": benchmark_coverage,
            }
        )
        rows.append(row)
    return rows


def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    parts = []
    grouped = df.groupby(group_cols, dropna=False)
    for value in value_cols:
        tmp = grouped[value].agg(["mean", "std", "count"]).reset_index()
        tmp["se"] = tmp["std"] / np.sqrt(tmp["count"].clip(lower=1))
        tmp["metric"] = value
        tmp = tmp.rename(columns={"mean": "value_mean", "std": "value_sd", "count": "n"})
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def ree_rows(
    distributed_df: pd.DataFrame,
    centralized_df: pd.DataFrame,
    match_cols: list[str],
    metadata_cols: list[str],
) -> pd.DataFrame:
    cent = centralized_df[match_cols + ["tau", "ql_censored"]].rename(columns={"ql_censored": "ql_centralized"})
    merged = distributed_df.merge(cent, on=match_cols + ["tau"], how="left")
    merged["ree"] = merged["ql_censored"] / merged["ql_centralized"]
    return merged[metadata_cols + ["tau", "ql_censored", "ql_centralized", "ql_ratio", "ree"]]
