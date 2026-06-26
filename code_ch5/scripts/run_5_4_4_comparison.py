#!/usr/bin/env python3
"""Run Section 5.4.4: DA-CSQRNN vs DAqrnn and optional DeepQuantreg."""

from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults, selected_settings
from common.config import CENTRAL_N_REP, CENTRAL_N_TEST, CENTRAL_N_TRAIN, right_censoring_settings
from common.experiments import load_hyperparameter_map, run_comparison_replication
from common.metrics import summarize
from common.storage import make_run_dir, save_csv, save_json


def print_setting_ratios(rows: list[dict[str, object]], label: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no metric rows")
        return

    metric_df = df.dropna(subset=["tau"]) if "tau" in df.columns else df
    if not metric_df.empty:
        table = (
            metric_df.groupby(["method", "tau"], dropna=False)[
                ["ql_ratio", "piw_ratio", "coverage", "benchmark_coverage"]
            ]
            .mean()
            .reset_index()
            .sort_values(["method", "tau"])
        )
        print(f"\n[{label}] mean ratios over {metric_df['rep'].nunique()} replication(s)")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    skipped = df[df.get("tau").isna()] if "tau" in df.columns else pd.DataFrame()
    if not skipped.empty:
        message_col = "error_message" if "error_message" in skipped.columns else "status"
        if message_col in skipped.columns:
            messages = skipped[message_col].dropna().astype(str).unique()
            if len(messages):
                print(f"[{label}] DeepQuantreg skipped/error: {messages[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None, help="Run only one simulation scenario.")
    parser.add_argument("--error", choices=["normal", "t3"], default=None, help="Run only one error distribution.")
    parser.add_argument("--censor-rate", type=float, choices=[0.25, 0.50], default=None, help="Run only one nominal censoring rate.")
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 200 for the coverage sensitivity run.")
    parser.add_argument(
        "--pi-method",
        choices=["iterative", "independent"],
        default="iterative",
        help="Benchmark PI method for DA-CSQRNN and DAqrnn: iterative warm-start or independent cold-start.",
    )
    parser.add_argument("--epsilon-huber", type=float, default=0.1, help="Huber smoothing parameter for the DAqrnn baseline.")
    parser.add_argument("--skip-dacsqrnn", action="store_true", help="Skip DA-CSQRNN rows in 5.4.4; reuse the 5.4.2 results instead.")
    parser.add_argument("--skip-daqrnn", action="store_true", help="Skip DAqrnn rows.")
    parser.add_argument("--skip-deepquantreg", action="store_true", default=False, help="Skip the external DeepQuantreg baseline.")
    parser.add_argument("--include-deepquantreg", dest="skip_deepquantreg", action="store_false", help="Run the external DeepQuantreg baseline.")
    parser.add_argument(
        "--only-deepquantreg",
        action="store_true",
        help="Run only DeepQuantreg rows, skipping DA-CSQRNN and DAqrnn.",
    )
    parser.add_argument(
        "--deepquantreg-bootstrap-reps",
        type=int,
        default=None,
        help="DeepQuantreg paired-bootstrap reps for PIW. Default matches the benchmark PI sample count.",
    )
    parser.add_argument(
        "--deepquantreg-pi-benchmark",
        choices=["own", "shared", "matched"],
        default="own",
        help="DeepQuantreg benchmark for PIW/coverage: own DeepQuantreg*, shared CS-QRNN*, or matched cold-start CS-QRNN*.",
    )
    args = apply_quick_defaults(parser.parse_args(), CENTRAL_N_TRAIN, CENTRAL_N_TEST, CENTRAL_N_REP)
    if args.only_deepquantreg:
        args.skip_dacsqrnn = True
        args.skip_daqrnn = True
        args.skip_deepquantreg = False

    run_dir = make_run_dir(args.out, "5_4_4_deepquanreg")
    raw_dir = run_dir / "raw"
    metric_rows = []
    fit_rows = []
    settings = list(selected_settings(right_censoring_settings(), args))
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
        J_dacsqrnn, lam_dacsqrnn = load_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            scenario=scenario,
            error=error,
            censor_type="right",
            censor_rate=censor_rate,
            method="DA-CSQRNN",
        )
        # No separate DAqrnn EBIC table is produced by Section 5.4.1, so the
        # comparison keeps the same censoring-specific J/lambda values.
        J_daqrnn, lam_daqrnn = J_dacsqrnn, lam_dacsqrnn

        for rep in tqdm(range(1, args.n_rep + 1), desc=f"S{scenario}/{error}/right/{censor_rate}", leave=False):
            rows, fits = run_comparison_replication(
                scenario,
                error,
                "right",
                censor_rate,
                args.n_train,
                args.n_test,
                rep,
                args.test_seed,
                args.seed,
                J_dacsqrnn,
                lam_dacsqrnn,
                J_daqrnn,
                lam_daqrnn,
                args.max_iter,
                args.bootstrap_reps,
                raw_dir,
                S=args.S,
                J_benchmark=J_benchmark,
                lam_benchmark=lam_benchmark,
                epsilon_huber=args.epsilon_huber,
                include_dacsqrnn=not args.skip_dacsqrnn,
                include_daqrnn=not args.skip_daqrnn,
                include_deepquantreg=not args.skip_deepquantreg,
                deepquantreg_bootstrap_reps=args.deepquantreg_bootstrap_reps,
                deepquantreg_pi_benchmark=args.deepquantreg_pi_benchmark,
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
            f"5.4.4 scenario={scenario}, error={error}, censor=right, rate={censor_rate:.0%}",
        )

    raw = save_csv(run_dir / "metrics_raw.csv", metric_rows)
    save_csv(run_dir / "fit_log.csv", fit_rows)
    summary_input = raw.dropna(subset=["tau"]) if "tau" in raw.columns else raw
    summary = summarize(
        summary_input,
        ["scenario", "error", "censor_rate", "method", "tau"],
        ["ql_ratio", "piw_ratio", "coverage", "benchmark_coverage"],
    )
    save_csv(run_dir / "summary.csv", summary)
    save_json(run_dir / "run_config.json", vars(args) | {"settings": [list(s) for s in settings]})
    print(f"saved comparison outputs to {run_dir}")


if __name__ == "__main__":
    main()
