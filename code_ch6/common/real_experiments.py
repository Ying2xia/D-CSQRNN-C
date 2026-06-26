"""Experiment orchestration for Chapter 6 real-data applications."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .augmentation import run_da_csqrnn
from .baselines import run_daqrnn_baseline, run_deepquantreg_adapter
from .data import SimDataset, as_fully_observed, make_tau_grid
from .distributed import partition_dataset, run_dcs_qrnn_c, run_one_shot
from .experiments import benchmark_predictions
from .metrics import interval_bounds, metric_rows_from_predictions, quantile_loss
from .real_config import CENTRAL_DA_ITERATIONS, TARGET_TAUS, distributed_da_iterations, fixed_tau_grid
from .real_data import (
    RealRawDataset,
    apply_real_censoring,
    censoring_level_summary,
    split_and_standardize,
)
from .training import fit_models_for_taus, tau_key


def load_real_hyperparameter_map(
    csv_path: str | Path | None,
    default_J: int,
    default_lambda: float,
    dataset: str | None = None,
    censor_type: str | None = None,
    censor_rate: float | None = None,
    method: str | None = None,
) -> tuple[dict[float, int], dict[float, float]]:
    """Load Chapter 6 per-tau hyperparameters when available."""

    J_map = {tau_key(t): int(default_J) for t in TARGET_TAUS}
    lam_map = {tau_key(t): float(default_lambda) for t in TARGET_TAUS}
    if not csv_path:
        return J_map, lam_map
    path = Path(csv_path).expanduser()
    if not path.exists():
        return J_map, lam_map
    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    filters = {
        "dataset": dataset,
        "censor_type": censor_type,
        "censor_rate": censor_rate,
        "method": method,
    }
    for col, value in filters.items():
        if value is not None and col in df.columns:
            mask &= df[col] == value
    sub = df[mask]
    for _, row in sub.iterrows():
        tau = tau_key(row["tau"])
        if "J" in row:
            J_map[tau] = int(row["J"])
        if "lambda" in row:
            lam_map[tau] = float(row["lambda"])
    return J_map, lam_map


def _point_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J,
    lam,
    rng: np.random.Generator,
    max_iter: int,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    taus = [tau_key(t) for t in target_taus]
    models, rows = fit_models_for_taus(
        X_train,
        y_train,
        taus,
        J,
        lam,
        rng,
        max_iter=max_iter,
        loss_kind=loss_kind,
        epsilon=epsilon,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    pred = np.column_stack([models[tau].predict(X_test) for tau in taus])
    return pred, rows


def _point_metric_rows(
    y_test: np.ndarray,
    target_taus: list[float],
    pred: np.ndarray,
    benchmark_pred: np.ndarray,
    metadata: dict[str, object],
    centralized_pred: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows = []
    for j, tau in enumerate(target_taus):
        ql = quantile_loss(y_test, pred[:, j], tau)
        ql_b = quantile_loss(y_test, benchmark_pred[:, j], tau)
        ql_central = np.nan
        ree = np.nan
        if centralized_pred is not None:
            ql_central = quantile_loss(y_test, centralized_pred[:, j], tau)
            ree = ql / ql_central if ql_central > 0 else np.nan
        row = dict(metadata)
        row.update(
            {
                "tau": tau_key(tau),
                "ql_censored": ql,
                "ql_benchmark": ql_b,
                "ql_ratio": ql / ql_b if ql_b > 0 else np.nan,
                "ql_centralized": ql_central,
                "ree": ree,
                "piw_censored": np.nan,
                "piw_benchmark": np.nan,
                "piw_ratio": np.nan,
                "coverage": np.nan,
                "benchmark_coverage": np.nan,
            }
        )
        rows.append(row)
    return rows


def run_real_centralized_replication(
    data: RealRawDataset,
    dataset_name: str,
    censor_type: str,
    censor_rate: float,
    rep: int,
    base_seed: int,
    J,
    lam,
    max_iter: int,
    S: int = CENTRAL_DA_ITERATIONS,
    bootstrap_reps: int = 0,
    J_benchmark=None,
    lam_benchmark=None,
    train_fraction: float = 0.8,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run one Boston/Gilgais DA-CSQRNN replication with QL and PIW ratios."""

    rng = np.random.default_rng(int(base_seed) + int(rep))
    split = split_and_standardize(data, rng, train_fraction=train_fraction)
    train = apply_real_censoring(split.X_train, split.y_train, dataset_name, censor_type, censor_rate, rng)
    tau_grid = make_tau_grid(train.n)
    target_taus = [tau_key(t) for t in TARGET_TAUS]
    bench_S = max(int(bootstrap_reps or 0), int(S))
    if J_benchmark is None:
        J_benchmark = J
    if lam_benchmark is None:
        lam_benchmark = lam

    benchmark_pred, benchmark_boot, bench_lower, bench_upper, bench_rows = benchmark_predictions(
        train,
        split.X_test,
        target_taus,
        J_benchmark,
        lam_benchmark,
        rng,
        bootstrap_reps=bench_S,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    da = run_da_csqrnn(
        train,
        split.X_test,
        target_taus,
        J,
        lam,
        S=int(S),
        tau_grid=tau_grid,
        rng=rng,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    meta = {
        "rep": rep,
        "dataset": dataset_name,
        "method": "DA-CSQRNN",
        "censor_type": censor_type,
        "censor_rate": float(censor_rate),
        "n_train": train.n,
        "n_test": int(split.X_test.shape[0]),
        "S": int(S),
        "benchmark_reps": bench_S,
        "actual_censor_rate": float((train.delta != 0).mean()),
        "benchmark_method": "CS-QRNN-star",
    }
    metric_rows = metric_rows_from_predictions(
        split.y_test,
        target_taus,
        da.final_predictions,
        benchmark_pred,
        da.lower,
        da.upper,
        bench_lower,
        bench_upper,
        meta,
        true_quantiles=None,
    )
    fit_rows = []
    for row in da.fit_rows + bench_rows:
        fit_rows.append({**row, **{k: meta[k] for k in ("rep", "dataset", "method", "censor_type", "censor_rate")}})
    censor_summary = {**meta, **censoring_level_summary(train)}
    return metric_rows, fit_rows, censor_summary


def run_real_comparison_replication(
    data: RealRawDataset,
    dataset_name: str,
    censor_type: str,
    censor_rate: float,
    rep: int,
    base_seed: int,
    J_dacsqrnn,
    lam_dacsqrnn,
    max_iter: int,
    S: int = CENTRAL_DA_ITERATIONS,
    J_daqrnn=None,
    lam_daqrnn=None,
    J_benchmark=None,
    lam_benchmark=None,
    J_huber_benchmark=None,
    lam_huber_benchmark=None,
    train_fraction: float = 0.8,
    epsilon_huber: float = 0.1,
    include_deepquantreg: bool = False,
    include_dacsqrnn: bool = True,
    include_daqrnn: bool = True,
    deepquantreg_dir: Path | None = None,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run one real-data comparison replication.

    The comparison table reports QL ratios only. Therefore this path fits
    point benchmarks and avoids the expensive bootstrap PI construction.
    """

    rng = np.random.default_rng(int(base_seed) + int(rep))
    split = split_and_standardize(data, rng, train_fraction=train_fraction)
    train = apply_real_censoring(split.X_train, split.y_train, dataset_name, censor_type, censor_rate, rng)
    tau_grid = make_tau_grid(train.n)
    target_taus = [tau_key(t) for t in TARGET_TAUS]
    if J_daqrnn is None:
        J_daqrnn = J_dacsqrnn
    if lam_daqrnn is None:
        lam_daqrnn = lam_dacsqrnn
    if J_benchmark is None:
        J_benchmark = J_dacsqrnn
    if lam_benchmark is None:
        lam_benchmark = lam_dacsqrnn
    if J_huber_benchmark is None:
        J_huber_benchmark = J_daqrnn
    if lam_huber_benchmark is None:
        lam_huber_benchmark = lam_daqrnn

    cs_bench_pred = None
    cs_bench_rows: list[dict[str, object]] = []
    if include_dacsqrnn:
        cs_bench_pred, cs_bench_rows = _point_predictions(
            train.X,
            train.y_true,
            split.X_test,
            target_taus,
            J_benchmark,
            lam_benchmark,
            rng,
            max_iter=max_iter,
            loss_kind="cs",
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
    huber_bench_pred = None
    huber_bench_rows: list[dict[str, object]] = []
    if include_daqrnn:
        huber_bench_pred, huber_bench_rows = _point_predictions(
            train.X,
            train.y_true,
            split.X_test,
            target_taus,
            J_huber_benchmark,
            lam_huber_benchmark,
            rng,
            max_iter=max_iter,
            loss_kind="huber",
            epsilon=epsilon_huber,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
    da_cs = None
    if include_dacsqrnn:
        da_cs = run_da_csqrnn(
            train,
            split.X_test,
            target_taus,
            J_dacsqrnn,
            lam_dacsqrnn,
            S=int(S),
            tau_grid=tau_grid,
            rng=rng,
            max_iter=max_iter,
            loss_kind="cs",
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
    da_huber = None
    if include_daqrnn:
        da_huber = run_daqrnn_baseline(
            train,
            split.X_test,
            target_taus,
            J_daqrnn,
            lam_daqrnn,
            S=int(S),
            tau_grid=tau_grid,
            rng=rng,
            max_iter=max_iter,
            epsilon=epsilon_huber,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
    base_meta = {
        "rep": rep,
        "dataset": dataset_name,
        "censor_type": censor_type,
        "censor_rate": float(censor_rate),
        "n_train": train.n,
        "n_test": int(split.X_test.shape[0]),
        "S": int(S),
        "actual_censor_rate": float((train.delta != 0).mean()),
    }
    metric_rows = []
    if da_cs is not None and cs_bench_pred is not None:
        metric_rows.extend(_point_metric_rows(
            split.y_test,
            target_taus,
            da_cs.final_predictions,
            cs_bench_pred,
            {**base_meta, "method": "DA-CSQRNN", "benchmark_method": "CS-QRNN-star"},
        ))
    if da_huber is not None and huber_bench_pred is not None:
        metric_rows.extend(_point_metric_rows(
            split.y_test,
            target_taus,
            da_huber.final_predictions,
            huber_bench_pred,
            {**base_meta, "method": "DAqrnn", "benchmark_method": "DAqrnn-star"},
        ))

    fit_rows = (
        ([] if da_cs is None else list(da_cs.fit_rows))
        + ([] if da_huber is None else list(da_huber.fit_rows))
        + list(cs_bench_rows)
        + list(huber_bench_rows)
    )
    if include_deepquantreg:
        full_train = as_fully_observed(train)
        deep_root = Path("deepquantreg_work") if deepquantreg_dir is None else Path(deepquantreg_dir)
        deep_dir = deep_root / f"{dataset_name}_{censor_type}_rep{rep:04d}"
        dq_pred = run_deepquantreg_adapter(train, split.X_test, target_taus, deep_dir / "point")
        dq_bench = run_deepquantreg_adapter(full_train, split.X_test, target_taus, deep_dir / "benchmark")
        metric_rows.extend(_point_metric_rows(
            split.y_test,
            target_taus,
            dq_pred,
            dq_bench,
            {**base_meta, "method": "DeepQuantreg", "benchmark_method": "DeepQuantreg-star"},
        ))
    fit_rows = [{**row, **base_meta} for row in fit_rows]
    return metric_rows, fit_rows, {**base_meta, **censoring_level_summary(train)}


def run_real_distributed_setting(
    data: RealRawDataset,
    censor_type: str,
    censor_rate: float,
    K_values: list[int],
    R: int,
    rep: int,
    base_seed: int,
    J,
    lam,
    max_iter: int,
    n_train: int,
    n_test: int | None,
    S: int | None = None,
    J_benchmark=None,
    lam_benchmark=None,
    tau_grid_size: int = 100,
    include_one_shot: bool = True,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run one household-power distributed real-data setting.

    Centralized DA-CSQRNN is fit once and reused for all K values so the REE
    denominator is identical across worker configurations.
    """

    rng = np.random.default_rng(int(base_seed) + int(rep))
    split = split_and_standardize(data, rng, n_train=n_train, n_test=n_test)
    train = apply_real_censoring(split.X_train, split.y_train, data.name, censor_type, censor_rate, rng)
    target_taus = [tau_key(t) for t in TARGET_TAUS]
    tau_grid = fixed_tau_grid(tau_grid_size)
    S = int(distributed_da_iterations(censor_rate) if S is None else S)
    if J_benchmark is None:
        J_benchmark = J
    if lam_benchmark is None:
        lam_benchmark = lam

    benchmark_pred, bench_rows = _point_predictions(
        train.X,
        train.y_true,
        split.X_test,
        target_taus,
        J_benchmark,
        lam_benchmark,
        rng,
        max_iter=max_iter,
        loss_kind="cs",
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = list(bench_rows)
    timing_rows: list[dict[str, object]] = []

    central_meta = {
        "rep": rep,
        "dataset": data.name,
        "method": "DA-CSQRNN",
        "K": np.nan,
        "R": np.nan,
        "censor_type": censor_type,
        "censor_rate": float(censor_rate),
        "n_train": train.n,
        "n_test": int(split.X_test.shape[0]),
        "S": S,
        "actual_censor_rate": float((train.delta != 0).mean()),
        "benchmark_method": "CS-QRNN-star",
    }
    t0 = time.perf_counter()
    central = run_da_csqrnn(
        train,
        split.X_test,
        target_taus,
        J,
        lam,
        S=S,
        tau_grid=tau_grid,
        rng=rng,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    central_time = time.perf_counter() - t0
    metric_rows.extend(_point_metric_rows(split.y_test, target_taus, central.final_predictions, benchmark_pred, central_meta))
    timing_rows.append({**central_meta, "time_seconds": central_time, "ract": 1.0})
    fit_rows.extend({**row, **central_meta} for row in central.fit_rows)

    for K in K_values:
        partitions = partition_dataset(train, int(K), rng)
        dcs = run_dcs_qrnn_c(
            partitions,
            split.X_test,
            target_taus,
            J,
            lam,
            S=S,
            R=int(R),
            tau_grid=tau_grid,
            rng=rng,
            max_iter=max_iter,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        meta = {
            **central_meta,
            "method": "DCS-QRNN-C",
            "K": int(K),
            "R": int(R),
        }
        metric_rows.extend(_point_metric_rows(
            split.y_test,
            target_taus,
            dcs.final_predictions,
            benchmark_pred,
            meta,
            centralized_pred=central.final_predictions,
        ))
        timing_rows.append({**meta, "time_seconds": dcs.elapsed_seconds, "ract": central_time / dcs.elapsed_seconds if dcs.elapsed_seconds > 0 else np.nan})
        fit_rows.extend({**row, **meta} for row in dcs.fit_rows)

        if include_one_shot:
            t0 = time.perf_counter()
            os_pred, os_rows = run_one_shot(
                partitions,
                split.X_test,
                target_taus,
                J,
                lam,
                S=S,
                tau_grid=tau_grid,
                rng=rng,
                max_iter=max_iter,
                backend=backend,
                device=device,
                torch_dtype=torch_dtype,
            )
            os_time = time.perf_counter() - t0
            os_meta = {**central_meta, "method": "OS", "K": int(K), "R": 0}
            metric_rows.extend(_point_metric_rows(
                split.y_test,
                target_taus,
                os_pred,
                benchmark_pred,
                os_meta,
                centralized_pred=central.final_predictions,
            ))
            timing_rows.append({**os_meta, "time_seconds": os_time, "ract": central_time / os_time if os_time > 0 else np.nan})
            fit_rows.extend({**row, **os_meta} for row in os_rows)

    return metric_rows, fit_rows, timing_rows, {**central_meta, **censoring_level_summary(train)}
