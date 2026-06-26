#!/usr/bin/env python3
"""Run Section 5.5.3: wall-clock time and RACT."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults
from common.config import DIST_N_REP, DIST_N_TEST, DIST_N_TRAIN, DIST_WORKERS
from common.distributed_reporting import print_setting_summary, save_progress, save_timing_paper_table
from common.distributed_experiments import run_distributed_replication
from common.experiments import load_hyperparameter_map
from common.storage import make_run_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--K", type=int, nargs="*", default=list(DIST_WORKERS))
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 20 for 25%% censoring and 50 for 50%%.")
    parser.add_argument("--include-one-shot", dest="include_one_shot", action="store_true", default=True)
    parser.add_argument("--skip-one-shot", dest="include_one_shot", action="store_false")
    args = parser.parse_args()
    # Section 5.5.3 reports wall-clock time and RACT. Force the NumPy CPU path
    # so the timing table reflects algorithmic cost rather than GPU batching,
    # CUDA scheduling, or contention on a single shared GPU.
    args.backend = "numpy"
    args.device = "cpu"
    args = apply_quick_defaults(args, DIST_N_TRAIN, DIST_N_TEST, DIST_N_REP)

    run_dir = make_run_dir(args.out, "5_5_3_distributed_efficiency")
    metric_rows = []
    timing_rows = []
    fit_rows = []
    J_map, lam_map = load_hyperparameter_map(args.hyperparams, args.J, args.lam, 1, "normal", "right", 0.25)
    total = args.n_rep * len(args.K)
    with tqdm(total=total, desc="Efficiency") as pbar:
        for K in args.K:
            setting_times = []
            for rep in range(1, args.n_rep + 1):
                rows, fits, times = run_distributed_replication(
                    1,
                    "normal",
                    "right",
                    0.25,
                    args.n_train,
                    args.n_test,
                    K,
                    args.R,
                    rep,
                    args.test_seed,
                    args.seed + 2000 * K,
                    J_map,
                    lam_map,
                    args.max_iter,
                    args.bootstrap_reps,
                    run_dir / "raw",
                    S=args.S,
                    include_centralized=True,
                    include_one_shot=args.include_one_shot,
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
                metric_rows.extend(rows)
                fit_rows.extend(fits)
                timing_rows.extend(times)
                setting_times.extend(times)
                pbar.set_postfix(rep=rep, K=K)
                pbar.update(1)
            print_setting_summary(
                setting_times,
                f"5.5.3 K={K}, R={args.R}",
                ["method", "K", "R"],
                ["time_seconds", "ract"],
            )
            save_progress(
                run_dir,
                metric_rows,
                fit_rows,
                timing_rows,
                timing_group_cols=["method", "K", "R"],
                timing_summary_name="summary.csv",
            )
            save_timing_paper_table(run_dir, timing_rows)

    save_json(run_dir / "run_config.json", vars(args))
    print(f"saved efficiency outputs to {run_dir}")


if __name__ == "__main__":
    main()
