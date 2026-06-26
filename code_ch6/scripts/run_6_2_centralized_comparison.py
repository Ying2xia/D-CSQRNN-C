#!/usr/bin/env python3
"""Run Chapter 6 centralized comparison with DAqrnn and DeepQuantreg."""

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
from common.real_experiments import load_real_hyperparameter_map, run_real_comparison_replication
from common.real_reporting import (
    print_metric_summary,
    save_central_comparison_tables,
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
    parser.add_argument("--train-fraction", type=float, default=CENTRAL_TRAIN_FRACTION)
    parser.add_argument("--epsilon-huber", type=float, default=0.1)
    parser.add_argument("--include-deepquantreg", dest="include_deepquantreg", action="store_true", default=True)
    parser.add_argument("--skip-deepquantreg", dest="include_deepquantreg", action="store_false")
    parser.add_argument(
        "--only-deepquantreg",
        action="store_true",
        help="Run only DeepQuantreg rows. DeepQuantreg supports right censoring only, so this forces --censor-types right.",
    )
    args = apply_runtime_defaults(parser.parse_args(), n_rep=CENTRAL_N_REP, max_iter=250)
    if args.quick:
        args.S = min(args.S, 2)
        args.datasets = args.datasets[:1]
        args.censor_types = args.censor_types[:1]
        args.include_deepquantreg = False
    if args.only_deepquantreg:
        args.censor_types = ["right"]
        args.include_deepquantreg = True

    run_dir = make_run_dir(args.out, "6_2_deepquanreg" if args.only_deepquantreg else "6_2_centralized_comparison")
    metric_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    censor_rows: list[dict[str, object]] = []

    datasets = {name: load_real_dataset(name, args.data_dir) for name in args.datasets}
    settings = [(name, censor_type) for name in args.datasets for censor_type in args.censor_types]
    for dataset_name, censor_type in tqdm(settings, desc="6.2 settings"):
        print(f"\nRunning 6.2 setting: dataset={dataset_name}, censor={censor_type}, rate={args.censor_rate:.0%}")
        J_cs, lam_cs = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            censor_type=censor_type,
            censor_rate=args.censor_rate,
            method="DA-CSQRNN",
        )
        J_daqrnn, lam_daqrnn = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            censor_type=censor_type,
            censor_rate=args.censor_rate,
            method="DAqrnn",
        )
        J_bench, lam_bench = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            method="CS-QRNN-star",
        )
        J_huber_bench, lam_huber_bench = load_real_hyperparameter_map(
            args.hyperparams,
            args.J,
            args.lam,
            dataset=dataset_name,
            method="DAqrnn-star",
        )
        setting_rows = []
        for rep in tqdm(range(1, args.n_rep + 1), desc=f"{dataset_name}/{censor_type}", leave=False):
            include_dq = bool(args.include_deepquantreg and censor_type == "right")
            try:
                rows, fits, censor = run_real_comparison_replication(
                    datasets[dataset_name],
                    dataset_name,
                    censor_type,
                    args.censor_rate,
                    rep,
                    args.seed,
                    J_cs,
                    lam_cs,
                    args.max_iter,
                    S=args.S,
                    J_daqrnn=J_daqrnn,
                    lam_daqrnn=lam_daqrnn,
                    J_benchmark=J_bench,
                    lam_benchmark=lam_bench,
                    J_huber_benchmark=J_huber_bench,
                    lam_huber_benchmark=lam_huber_bench,
                    train_fraction=args.train_fraction,
                    epsilon_huber=args.epsilon_huber,
                    include_deepquantreg=include_dq,
                    include_dacsqrnn=not args.only_deepquantreg,
                    include_daqrnn=not args.only_deepquantreg,
                    deepquantreg_dir=run_dir / "raw" / "deepquantreg",
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
            except Exception as exc:
                if include_dq:
                    if args.only_deepquantreg:
                        raise
                    print(f"[6.2 {dataset_name}/{censor_type} rep={rep}] DeepQuantreg skipped/error: {exc}")
                    rows, fits, censor = run_real_comparison_replication(
                        datasets[dataset_name],
                        dataset_name,
                        censor_type,
                        args.censor_rate,
                        rep,
                        args.seed,
                        J_cs,
                        lam_cs,
                        args.max_iter,
                        S=args.S,
                        J_daqrnn=J_daqrnn,
                        lam_daqrnn=lam_daqrnn,
                        J_benchmark=J_bench,
                        lam_benchmark=lam_bench,
                        J_huber_benchmark=J_huber_bench,
                        lam_huber_benchmark=lam_huber_bench,
                        train_fraction=args.train_fraction,
                        epsilon_huber=args.epsilon_huber,
                        include_deepquantreg=False,
                        include_dacsqrnn=True,
                        include_daqrnn=True,
                        backend=args.backend,
                        device=args.device,
                        torch_dtype=args.torch_dtype,
                    )
                else:
                    raise
            metric_rows.extend(rows)
            setting_rows.extend(rows)
            fit_rows.extend(fits)
            censor_rows.append(censor)
        print_metric_summary(
            setting_rows,
            f"6.2 {dataset_name}/{censor_type}/{args.censor_rate:.0%}",
            ["method", "tau"],
            ["ql_ratio"],
        )
        save_progress(
            run_dir,
            metric_rows,
            fit_rows,
            censor_rows=censor_rows,
            metric_group_cols=["dataset", "censor_type", "censor_rate", "method", "tau"],
            metric_value_cols=["ql_ratio"],
        )
        save_central_comparison_tables(run_dir, metric_rows)

    save_json(run_dir / "run_config.json", vars(args) | {"settings": settings})
    print(f"saved Chapter 6 centralized comparison outputs to {run_dir}")


if __name__ == "__main__":
    main()
