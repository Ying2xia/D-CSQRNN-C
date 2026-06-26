#!/usr/bin/env python3
"""Run DAqrnn for left- and interval-censored Section 5.4.4 comparisons."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults, selected_settings
from common.baselines import run_daqrnn_baseline
from common.config import CENTRAL_N_REP, CENTRAL_N_TEST, CENTRAL_N_TRAIN, central_settings, da_iterations
from common.data import make_censored_dataset, make_tau_grid, make_uncensored_dataset, true_conditional_quantiles
from common.experiments import benchmark_predictions, load_hyperparameter_map
from common.metrics import metric_rows_from_predictions, summarize
from common.storage import make_run_dir, save_csv, save_json, save_npz
from common.training import tau_key


def print_setting_ratios(rows: list[dict[str, object]], label: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no metric rows")
        return
    table = (
        df.groupby("tau", dropna=False)[["ql_ratio", "piw_ratio", "coverage_c", "coverage_u"]]
        .mean()
        .reset_index()
    )
    print(f"\n[{label}] DAqrnn mean ratios over {df['rep'].nunique()} replication(s)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run_daqrnn_replication(
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    n_train: int,
    n_test: int,
    rep: int,
    test_seed: int,
    base_seed: int,
    J_daqrnn,
    lam_daqrnn,
    max_iter: int,
    bootstrap_reps: int,
    raw_dir,
    S: int | None,
    epsilon_huber: float,
    pi_method: str,
    backend: str,
    device: str,
    torch_dtype: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(base_seed + rep)
    test_rng = np.random.default_rng(test_seed)
    train = make_censored_dataset(n_train, scenario, error, censor_type, censor_rate, rng)
    test = make_uncensored_dataset(n_test, scenario, error, test_rng)
    tau_grid = make_tau_grid(n_train)
    target_taus = [tau_key(t) for t in (0.10, 0.25, 0.50, 0.75, 0.90)]
    S = int(da_iterations(censor_rate) if S is None else S)
    bench_S = max(int(bootstrap_reps or 0), S)
    true_q = true_conditional_quantiles(test.X, scenario, error, target_taus)

    # DAqrnn should be compared with its own fully observed Huber benchmark
    # rather than the CS-QRNN* benchmark used by DA-CSQRNN.
    benchmark_pred, benchmark_boot, bench_lower, bench_upper, bench_rows = benchmark_predictions(
        train,
        test.X,
        target_taus,
        J_daqrnn,
        lam_daqrnn,
        rng,
        bootstrap_reps=bench_S,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
        pi_method=pi_method,
        loss_kind="huber",
        epsilon=epsilon_huber,
    )
    da_huber = run_daqrnn_baseline(
        train,
        test.X,
        target_taus,
        J_daqrnn,
        lam_daqrnn,
        S=S,
        tau_grid=tau_grid,
        rng=rng,
        max_iter=max_iter,
        epsilon=epsilon_huber,
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
        "method": "DAqrnn",
        "n_train": n_train,
        "n_test": n_test,
        "S": S,
        "benchmark_reps": bench_S,
        "actual_censor_rate": float((train.delta != 0).mean()),
        "pi_benchmark": pi_method,
        "benchmark_method": "DAqrnn-star",
    }
    metric_rows = metric_rows_from_predictions(
        test.y_true,
        target_taus,
        da_huber.final_predictions,
        benchmark_pred,
        da_huber.lower,
        da_huber.upper,
        bench_lower,
        bench_upper,
        meta,
        true_quantiles=true_q,
    )
    for row in metric_rows:
        # Thesis table names: c = censored DAqrnn, u = uncensored benchmark.
        row["coverage_c"] = row["coverage"]
        row["coverage_u"] = row["benchmark_coverage"]

    save_npz(
        raw_dir / f"daqrnn_rep_{rep:04d}_s{scenario}_{error}_{censor_type}_{int(censor_rate * 100)}.npz",
        X_test=test.X,
        y_test=test.y_true,
        X_train=train.X,
        y_train_true=train.y_true,
        y_train_obs=train.y_obs,
        delta=train.delta,
        L=train.L,
        R=train.R,
        target_taus=np.array(target_taus),
        true_quantiles=true_q,
        tau_grid=tau_grid,
        benchmark_reps=np.array(bench_S),
        pred_daqrnn=da_huber.final_predictions,
        pred_daqrnn_iter=da_huber.iteration_predictions,
        lower_daqrnn=da_huber.lower,
        upper_daqrnn=da_huber.upper,
        pred_benchmark_daqrnn=benchmark_pred,
        pred_benchmark_daqrnn_boot=benchmark_boot,
        lower_benchmark_daqrnn=bench_lower,
        upper_benchmark_daqrnn=bench_upper,
    )

    fit_rows: list[dict[str, object]] = []
    for row in da_huber.fit_rows:
        out = dict(row)
        out.update(meta)
        out["fit_source"] = "DAqrnn"
        fit_rows.append(out)
    for row in bench_rows:
        out = dict(row)
        out.update(meta)
        out["fit_source"] = "DAqrnn-star"
        fit_rows.append(out)
    return metric_rows, fit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None, help="Run only one simulation scenario.")
    parser.add_argument("--error", choices=["normal", "t3"], default=None, help="Run only one error distribution.")
    parser.add_argument(
        "--censor-type",
        choices=["left", "interval"],
        default=None,
        help="Run only one censoring mechanism. This script only supports left and interval censoring.",
    )
    parser.add_argument("--censor-rate", type=float, choices=[0.25, 0.50], default=None, help="Run only one nominal censoring rate.")
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 200 for the coverage sensitivity run.")
    parser.add_argument(
        "--pi-method",
        choices=["iterative", "independent"],
        default="iterative",
        help="DAqrnn-star PI method: iterative warm-start or independent cold-start.",
    )
    parser.add_argument("--epsilon-huber", type=float, default=0.1, help="Huber smoothing parameter for DAqrnn.")
    args = apply_quick_defaults(parser.parse_args(), CENTRAL_N_TRAIN, CENTRAL_N_TEST, CENTRAL_N_REP)

    run_dir = make_run_dir(args.out, "5_4_4_daqrnn_left_interval")
    raw_dir = run_dir / "raw"
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    base_settings = [s for s in central_settings() if s[2] in {"left", "interval"}]
    settings = list(selected_settings(base_settings, args))
    for scenario, error, censor_type, censor_rate in tqdm(settings, desc="Settings"):
        setting_rows: list[dict[str, object]] = []
        # No separate DAqrnn EBIC table is produced, so use the same
        # censoring-specific hyperparameters selected for DA-CSQRNN.
        J_daqrnn, lam_daqrnn = load_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            scenario=scenario,
            error=error,
            censor_type=censor_type,
            censor_rate=censor_rate,
            method="DA-CSQRNN",
        )
        for rep in tqdm(
            range(1, args.n_rep + 1),
            desc=f"S{scenario}/{error}/{censor_type}/{censor_rate}",
            leave=False,
        ):
            rows, fits = run_daqrnn_replication(
                scenario,
                error,
                censor_type,
                censor_rate,
                args.n_train,
                args.n_test,
                rep,
                args.test_seed,
                args.seed,
                J_daqrnn,
                lam_daqrnn,
                args.max_iter,
                args.bootstrap_reps,
                raw_dir,
                args.S,
                args.epsilon_huber,
                args.pi_method,
                args.backend,
                args.device,
                args.torch_dtype,
            )
            metric_rows.extend(rows)
            setting_rows.extend(rows)
            fit_rows.extend(fits)
        print_setting_ratios(
            setting_rows,
            f"5.4.4 DAqrnn scenario={scenario}, error={error}, censor={censor_type}, rate={censor_rate:.0%}",
        )

    raw = save_csv(run_dir / "metrics_raw.csv", metric_rows)
    save_csv(run_dir / "fit_log.csv", fit_rows)
    group_cols = ["scenario", "error", "censor_type", "censor_rate", "method", "tau"]
    summary = summarize(raw, group_cols, ["ql_ratio", "piw_ratio", "coverage_c", "coverage_u"])
    save_csv(run_dir / "summary.csv", summary)
    paper_table = (
        raw.groupby(group_cols, dropna=False)[["ql_ratio", "piw_ratio", "coverage_c", "coverage_u"]]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )
    save_csv(run_dir / "paper_table_daqrnn_left_interval.csv", paper_table)
    save_json(run_dir / "run_config.json", vars(args) | {"settings": [list(s) for s in settings]})
    print(f"saved DAqrnn left/interval outputs to {run_dir}")


if __name__ == "__main__":
    main()
