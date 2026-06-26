#!/usr/bin/env python3
"""Run Chapter 6 distributed real-data experiment on household power data."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from _common import ROOT, add_common_args, apply_runtime_defaults
from common.real_config import (
    CENSOR_RATES,
    CENSOR_TYPES,
    HOUSEHOLD_ILEA_ROUNDS,
    HOUSEHOLD_N_REP,
    HOUSEHOLD_N_TRAIN,
    HOUSEHOLD_TAU_GRID_SIZE,
    HOUSEHOLD_WORKERS,
)
from common.real_data import load_real_dataset
from common.real_experiments import load_real_hyperparameter_map, run_real_distributed_setting
from common.real_reporting import (
    print_metric_summary,
    print_timing_summary,
    save_distributed_tables,
    save_progress,
)
from common.storage import make_run_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--data-dir", default=str(ROOT), help="Directory containing household_power_consumption.txt.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap while loading household data.")
    parser.add_argument("--n-train", type=int, default=HOUSEHOLD_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=None, help="Optional test row cap; default uses all remaining complete rows.")
    parser.add_argument("--K", type=int, nargs="*", default=list(HOUSEHOLD_WORKERS))
    parser.add_argument("--R", type=int, default=HOUSEHOLD_ILEA_ROUNDS)
    parser.add_argument("--censor-types", nargs="*", choices=list(CENSOR_TYPES), default=list(CENSOR_TYPES))
    parser.add_argument("--censor-rates", type=float, nargs="*", choices=list(CENSOR_RATES), default=list(CENSOR_RATES))
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 20 for 25%% and 50 for 50%%.")
    parser.add_argument("--tau-grid-size", type=int, default=HOUSEHOLD_TAU_GRID_SIZE)
    parser.add_argument("--include-one-shot", dest="include_one_shot", action="store_true", default=True)
    parser.add_argument("--skip-one-shot", dest="include_one_shot", action="store_false")
    args = apply_runtime_defaults(parser.parse_args(), n_rep=HOUSEHOLD_N_REP, max_iter=120)
    if args.quick:
        args.max_rows = args.max_rows or 3000
        args.n_train = min(args.n_train, 1000)
        args.n_test = args.n_test or 500
        args.K = [2]
        args.censor_types = args.censor_types[:1]
        args.censor_rates = args.censor_rates[:1]
        args.S = 2
        args.tau_grid_size = min(args.tau_grid_size, 20)

    run_dir = make_run_dir(args.out, "6_3_distributed_real")
    data = load_real_dataset("household_power", args.data_dir, max_rows=args.max_rows)
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    censor_rows: list[dict[str, object]] = []

    settings = [(censor_type, censor_rate) for censor_rate in args.censor_rates for censor_type in args.censor_types]
    for censor_type, censor_rate in tqdm(settings, desc="6.3 settings"):
        print(f"\nRunning 6.3 setting: dataset=household_power, censor={censor_type}, rate={censor_rate:.0%}")
        J_map, lam_map = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset="household_power",
            censor_type=censor_type,
            censor_rate=censor_rate,
            method="DCS-QRNN-C",
        )
        J_benchmark, lam_benchmark = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset="household_power",
            method="CS-QRNN-star",
        )
        setting_rows = []
        setting_times = []
        for rep in range(1, args.n_rep + 1):
            rows, fits, times, censor = run_real_distributed_setting(
                data,
                censor_type,
                censor_rate,
                K_values=list(args.K),
                R=args.R,
                rep=rep,
                base_seed=args.seed,
                J=J_map,
                lam=lam_map,
                max_iter=args.max_iter,
                n_train=args.n_train,
                n_test=args.n_test,
                S=args.S,
                J_benchmark=J_benchmark,
                lam_benchmark=lam_benchmark,
                tau_grid_size=args.tau_grid_size,
                include_one_shot=args.include_one_shot,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
            metric_rows.extend(rows)
            setting_rows.extend(rows)
            fit_rows.extend(fits)
            timing_rows.extend(times)
            setting_times.extend(times)
            censor_rows.append(censor)
        print_metric_summary(
            setting_rows,
            f"6.3 household_power/{censor_type}/{censor_rate:.0%}",
            ["method", "K", "tau"],
            ["ql_ratio", "ree"],
        )
        print_timing_summary(setting_times, f"6.3 timing household_power/{censor_type}/{censor_rate:.0%}")
        save_progress(
            run_dir,
            metric_rows,
            fit_rows,
            timing_rows,
            censor_rows,
            metric_group_cols=["dataset", "censor_type", "censor_rate", "method", "K", "R", "tau"],
            metric_value_cols=["ql_ratio", "ree"],
        )
        save_distributed_tables(run_dir, metric_rows, timing_rows)

    save_json(run_dir / "run_config.json", vars(args) | {"settings": settings})
    print(f"saved Chapter 6 distributed real-data outputs to {run_dir}")


if __name__ == "__main__":
    main()
