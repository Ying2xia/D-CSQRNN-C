#!/usr/bin/env python3
"""Run Chapter 6 centralized real-data DA-CSQRNN tables."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from _common import ROOT, add_common_args, apply_runtime_defaults
from common.real_config import (
    CENTRAL_CENSOR_RATE,
    CENTRAL_DA_ITERATIONS,
    CENTRAL_DATASETS,
    CENTRAL_N_REP,
    CENTRAL_TRAIN_FRACTION,
    CENSOR_TYPES,
)
from common.real_data import load_real_dataset
from common.real_experiments import load_real_hyperparameter_map, run_real_centralized_replication
from common.real_reporting import (
    print_metric_summary,
    save_central_main_tables,
    save_progress,
)
from common.storage import make_run_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--data-dir", default=str(ROOT), help="Directory containing local real-data files and cache.")
    parser.add_argument("--datasets", nargs="*", default=list(CENTRAL_DATASETS), choices=list(CENTRAL_DATASETS))
    parser.add_argument("--censor-types", nargs="*", default=list(CENSOR_TYPES), choices=list(CENSOR_TYPES))
    parser.add_argument("--censor-rate", type=float, default=CENTRAL_CENSOR_RATE)
    parser.add_argument("--S", type=int, default=CENTRAL_DA_ITERATIONS, help="DA iterations; thesis default is 200.")
    parser.add_argument("--bootstrap-reps", type=int, default=0, help="Minimum benchmark PI samples; 0 means use S.")
    parser.add_argument("--train-fraction", type=float, default=CENTRAL_TRAIN_FRACTION)
    args = apply_runtime_defaults(parser.parse_args(), n_rep=CENTRAL_N_REP, max_iter=250)
    if args.quick:
        args.S = min(args.S, 2)
        args.datasets = args.datasets[:1]
        args.censor_types = args.censor_types[:1]

    run_dir = make_run_dir(args.out, "6_1_centralized_real")
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    censor_rows: list[dict[str, object]] = []

    datasets = {name: load_real_dataset(name, args.data_dir) for name in args.datasets}
    settings = [(name, censor_type) for name in args.datasets for censor_type in args.censor_types]
    for dataset_name, censor_type in tqdm(settings, desc="6.1 settings"):
        print(f"\nRunning 6.1 setting: dataset={dataset_name}, censor={censor_type}, rate={args.censor_rate:.0%}")
        J_map, lam_map = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            censor_type=censor_type,
            censor_rate=args.censor_rate,
            method="DA-CSQRNN",
        )
        J_benchmark, lam_benchmark = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            method="CS-QRNN-star",
        )
        setting_rows = []
        for rep in tqdm(range(1, args.n_rep + 1), desc=f"{dataset_name}/{censor_type}", leave=False):
            rows, fits, censor = run_real_centralized_replication(
                datasets[dataset_name],
                dataset_name,
                censor_type,
                args.censor_rate,
                rep,
                args.seed,
                J_map,
                lam_map,
                args.max_iter,
                S=args.S,
                bootstrap_reps=args.bootstrap_reps,
                J_benchmark=J_benchmark,
                lam_benchmark=lam_benchmark,
                train_fraction=args.train_fraction,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
            metric_rows.extend(rows)
            setting_rows.extend(rows)
            fit_rows.extend(fits)
            censor_rows.append(censor)
        print_metric_summary(
            setting_rows,
            f"6.1 {dataset_name}/{censor_type}/{args.censor_rate:.0%}",
            ["tau"],
            ["ql_ratio", "piw_ratio"],
        )
        save_progress(
            run_dir,
            metric_rows,
            fit_rows,
            censor_rows=censor_rows,
            metric_group_cols=["dataset", "censor_type", "censor_rate", "method", "tau"],
            metric_value_cols=["ql_ratio", "piw_ratio"],
        )
        save_central_main_tables(run_dir, metric_rows)

    save_json(run_dir / "run_config.json", vars(args) | {"settings": settings})
    print(f"saved Chapter 6 centralized real-data outputs to {run_dir}")


if __name__ == "__main__":
    main()
