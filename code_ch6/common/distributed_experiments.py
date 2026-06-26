"""Shared orchestration for Section 5.5 distributed simulations."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .augmentation import run_da_csqrnn
from .config import TARGET_TAUS, distributed_da_iterations
from .data import make_censored_dataset, make_tau_grid, make_uncensored_dataset, true_conditional_quantiles
from .distributed import partition_dataset, run_dcs_qrnn_c, run_one_shot
from .experiments import benchmark_predictions
from .metrics import metric_rows_from_predictions, quantile_loss
from .storage import save_npz
from .training import tau_key


def run_distributed_replication(
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    n_train: int,
    n_test: int,
    K: int,
    R: int,
    rep: int,
    test_seed: int,
    base_seed: int,
    J,
    lam,
    max_iter: int,
    bootstrap_reps: int,
    raw_dir: Path,
    S: int | None = None,
    include_centralized: bool = True,
    include_one_shot: bool = False,
    target_taus=TARGET_TAUS,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    rng = np.random.default_rng(base_seed + rep)
    test_rng = np.random.default_rng(test_seed)
    train = make_censored_dataset(n_train, scenario, error, censor_type, censor_rate, rng)
    test = make_uncensored_dataset(n_test, scenario, error, test_rng)
    tau_grid = make_tau_grid(n_train)
    S = int(distributed_da_iterations(censor_rate) if S is None else S)
    target_taus = [tau_key(t) for t in target_taus]
    true_q = true_conditional_quantiles(test.X, scenario, error, target_taus)
    bench_S = max(int(bootstrap_reps or 0), S)

    benchmark_pred, benchmark_boot, bench_lower, bench_upper, bench_rows = benchmark_predictions(
        train,
        test.X,
        target_taus,
        J,
        lam,
        rng,
        bootstrap_reps=bench_S,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )

    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    central_da = None
    central_time = np.nan
    if include_centralized:
        t0 = time.perf_counter()
        central_da = run_da_csqrnn(
            train,
            test.X,
            target_taus,
            J,
            lam,
            S,
            tau_grid,
            rng,
            max_iter=max_iter,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        central_time = time.perf_counter() - t0
        meta = {
            "rep": rep,
            "scenario": scenario,
            "error": error,
            "censor_type": censor_type,
            "censor_rate": censor_rate,
            "method": "DA-CSQRNN",
            "K": np.nan,
            "R": np.nan,
            "n_train": n_train,
            "n_test": n_test,
            "S": S,
            "benchmark_reps": bench_S,
            "actual_censor_rate": float((train.delta != 0).mean()),
            "benchmark_method": "CS-QRNN-star",
        }
        metric_rows.extend(
            metric_rows_from_predictions(
                test.y_true,
                target_taus,
                central_da.final_predictions,
                benchmark_pred,
                central_da.lower,
                central_da.upper,
                bench_lower,
                bench_upper,
                meta,
                true_quantiles=true_q,
            )
        )
        timing_rows.append({**meta, "time_seconds": central_time, "ract": 1.0})
        fit_rows.extend({**row, **{k: meta[k] for k in ("rep", "scenario", "error", "censor_type", "censor_rate", "method")}} for row in central_da.fit_rows)

    partitions = partition_dataset(train, K, rng)
    dcs = run_dcs_qrnn_c(
        partitions,
        test.X,
        target_taus,
        J,
        lam,
        S,
        R,
        tau_grid,
        rng,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    meta = {
        "rep": rep,
        "scenario": scenario,
        "error": error,
        "censor_type": censor_type,
        "censor_rate": censor_rate,
        "method": "DCS-QRNN-C",
        "K": K,
        "R": R,
        "n_train": n_train,
        "n_test": n_test,
        "S": S,
        "benchmark_reps": bench_S,
        "actual_censor_rate": float((train.delta != 0).mean()),
        "benchmark_method": "CS-QRNN-star",
    }
    dcs_rows = metric_rows_from_predictions(
        test.y_true,
        target_taus,
        dcs.final_predictions,
        benchmark_pred,
        dcs.lower,
        dcs.upper,
        bench_lower,
        bench_upper,
        meta,
        true_quantiles=true_q,
    )
    if central_da is not None:
        for row in dcs_rows:
            tau = row["tau"]
            j = target_taus.index(tau)
            ql_central = quantile_loss(test.y_true, central_da.final_predictions[:, j], tau)
            row["ql_centralized"] = ql_central
            row["ree"] = row["ql_censored"] / ql_central if ql_central > 0 else np.nan
    metric_rows.extend(dcs_rows)
    ract = central_time / dcs.elapsed_seconds if np.isfinite(central_time) and dcs.elapsed_seconds > 0 else np.nan
    timing_rows.append({**meta, "time_seconds": dcs.elapsed_seconds, "ract": ract})
    fit_rows.extend({**row, **{k: meta[k] for k in ("rep", "scenario", "error", "censor_type", "censor_rate", "method", "K", "R")}} for row in dcs.fit_rows)

    one_shot_pred = None
    if include_one_shot:
        t0 = time.perf_counter()
        one_shot_pred, os_rows = run_one_shot(
            partitions,
            test.X,
            target_taus,
            J,
            lam,
            S,
            tau_grid,
            rng,
            max_iter=max_iter,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        os_time = time.perf_counter() - t0
        for j, tau in enumerate(target_taus):
            ql_os = quantile_loss(test.y_true, one_shot_pred[:, j], tau)
            ql_b = quantile_loss(test.y_true, benchmark_pred[:, j], tau)
            ql_central = (
                quantile_loss(test.y_true, central_da.final_predictions[:, j], tau)
                if central_da is not None
                else np.nan
            )
            metric_rows.append(
                {
                    "rep": rep,
                    "scenario": scenario,
                    "error": error,
                    "censor_type": censor_type,
                    "censor_rate": censor_rate,
                    "method": "OS",
                    "K": K,
                    "R": 0,
                    "n_train": n_train,
                    "n_test": n_test,
                    "S": S,
                    "benchmark_reps": bench_S,
                    "actual_censor_rate": float((train.delta != 0).mean()),
                    "tau": tau,
                    "ql_censored": ql_os,
                    "ql_benchmark": ql_b,
                    "ql_ratio": ql_os / ql_b if ql_b > 0 else np.nan,
                    "ql_centralized": ql_central,
                    "ree": ql_os / ql_central if ql_central > 0 else np.nan,
                    "piw_censored": np.nan,
                    "piw_benchmark": np.nan,
                    "piw_ratio": np.nan,
                    "coverage": np.nan,
                    "benchmark_coverage": np.nan,
                }
            )
        timing_rows.append(
            {
                "rep": rep,
                "scenario": scenario,
                "error": error,
                "censor_type": censor_type,
                "censor_rate": censor_rate,
                "method": "OS",
                "K": K,
                "R": 0,
                "n_train": n_train,
                "n_test": n_test,
                "S": S,
                "benchmark_reps": bench_S,
                "actual_censor_rate": float((train.delta != 0).mean()),
                "time_seconds": os_time,
                "ract": central_time / os_time if np.isfinite(central_time) and os_time > 0 else np.nan,
            }
        )
        fit_rows.extend(os_rows)

    save_npz(
        raw_dir / f"rep_{rep:04d}_s{scenario}_{error}_{censor_type}_{int(censor_rate*100)}_K{K}_R{R}.npz",
        X_test=test.X,
        y_test=test.y_true,
        X_train=train.X,
        y_train_true=train.y_true,
        y_train_obs=train.y_obs,
        delta=train.delta,
        L=train.L,
        R_boundary=train.R,
        target_taus=np.array(target_taus),
        true_quantiles=true_q,
        tau_grid=tau_grid,
        benchmark_reps=np.array(bench_S),
        pred_benchmark=benchmark_pred,
        pred_benchmark_boot=benchmark_boot,
        lower_benchmark=bench_lower,
        upper_benchmark=bench_upper,
        pred_central=np.array([]) if central_da is None else central_da.final_predictions,
        pred_central_iter=np.array([]) if central_da is None else central_da.iteration_predictions,
        lower_central=np.array([]) if central_da is None else central_da.lower,
        upper_central=np.array([]) if central_da is None else central_da.upper,
        pred_dcs=dcs.final_predictions,
        pred_dcs_iter=dcs.iteration_predictions,
        lower_dcs=dcs.lower,
        upper_dcs=dcs.upper,
        pred_one_shot=np.array([]) if one_shot_pred is None else one_shot_pred,
        tau_schedule_dcs=dcs.tau_schedule,
    )
    fit_rows.extend(bench_rows)
    return metric_rows, fit_rows, timing_rows
