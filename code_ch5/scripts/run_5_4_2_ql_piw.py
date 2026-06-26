#!/usr/bin/env python3
"""Run Section 5.4.2: QL ratio and PIW ratio for DA-CSQRNN."""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from _common import add_common_args, add_filter_args, apply_quick_defaults, selected_settings
from common.config import CENTRAL_N_REP, CENTRAL_N_TEST, CENTRAL_N_TRAIN, central_settings
from common.experiments import load_hyperparameter_map, run_centralized_replication
from common.metrics import summarize
from common.storage import make_run_dir, save_csv, save_json


def print_setting_ratios(rows: list[dict[str, object]], label: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no metric rows")
        return
    cols = ["ql_ratio", "piw_ratio", "coverage", "benchmark_coverage"]
    table = df.groupby("tau", dropna=False)[cols].mean().reset_index()
    print(f"\n[{label}] mean ratios over {df['rep'].nunique()} replication(s)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_filter_args(parser)
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 200 for the coverage sensitivity run.")
    parser.add_argument(
        "--pi-method",
        choices=["iterative", "independent"],
        default="iterative",
        help="Benchmark PI method: iterative is warm-started; independent is ordinary cold-start bootstrap for denominator sensitivity.",
    )
    args = apply_quick_defaults(parser.parse_args(), CENTRAL_N_TRAIN, CENTRAL_N_TEST, CENTRAL_N_REP)

    run_dir = make_run_dir(args.out, "5_4_2_ql_piw")
    raw_dir = run_dir / "raw"
    metric_rows = []
    fit_rows = []
    settings = list(selected_settings(central_settings(), args))
    for scenario, error, censor_type, censor_rate in tqdm(settings, desc="Settings"):
        setting_rows = []
        J_benchmark, lam_benchmark = load_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            scenario=scenario,
            error=error,
            method="CS-QRNN-star",
        )
        J_map, lam_map = load_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            scenario=scenario,
            error=error,
            censor_type=censor_type,
            censor_rate=censor_rate,
            method="DA-CSQRNN",
        )
        for rep in tqdm(range(1, args.n_rep + 1), desc=f"S{scenario}/{error}/{censor_type}/{censor_rate}", leave=False):
            rows, fits = run_centralized_replication(
                scenario,
                error,
                censor_type,
                censor_rate,
                args.n_train,
                args.n_test,
                rep,
                args.test_seed,
                args.seed,
                J_map,
                lam_map,
                args.max_iter,
                args.bootstrap_reps,
                raw_dir,
                S=args.S,
                J_benchmark=J_benchmark,
                lam_benchmark=lam_benchmark,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
                pi_method=args.pi_method,
            )
            metric_rows.extend(rows)
            setting_rows.extend(rows)
            fit_rows.extend(fits)
        print_setting_ratios(
            setting_rows,
            f"5.4.2 scenario={scenario}, error={error}, censor={censor_type}, rate={censor_rate:.0%}",
        )

    raw = save_csv(run_dir / "metrics_raw.csv", metric_rows)
    save_csv(run_dir / "fit_log.csv", fit_rows)
    group_cols = ["scenario", "error", "censor_type", "censor_rate", "tau"]
    summary = summarize(
        raw,
        group_cols,
        [
            "ql_censored",
            "ql_benchmark",
            "ql_ratio",
            "piw_censored",
            "piw_benchmark",
            "piw_ratio",
            "coverage",
            "benchmark_coverage",
        ],
    )
    save_csv(run_dir / "summary.csv", summary)
    save_json(
        run_dir / "run_config.json",
        vars(args) | {"settings": [list(s) for s in settings]},
    )
    print(f"saved raw metrics and summaries to {run_dir}")


if __name__ == "__main__":
    main()
