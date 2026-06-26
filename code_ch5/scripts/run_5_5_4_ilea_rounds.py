#!/usr/bin/env python3
"""Run Section 5.5.4: sensitivity to the number of ILEA rounds."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults
from common.config import DIST_N_REP, DIST_N_TEST, DIST_N_TRAIN, ILEA_ROUNDS_GRID
from common.distributed_reporting import print_setting_summary, save_progress
from common.distributed_experiments import run_distributed_replication
from common.experiments import load_hyperparameter_map
from common.metrics import quantile_loss, summarize
from common.storage import make_run_dir, save_csv, save_json


def _ilea_iteration_rows(raw_path, rep: int, R: int, tau: float) -> list[dict[str, object]]:
    data = np.load(raw_path)
    y = data["y_test"]
    pred_benchmark = data["pred_benchmark"][:, 0]
    pred_central = data["pred_central"][:, 0]
    dcs_iter = data["pred_dcs_iter"][:, :, 0]
    ql_benchmark = quantile_loss(y, pred_benchmark, tau)
    ql_central = quantile_loss(y, pred_central, tau)

    rows = []
    phases = [("first", dcs_iter[0])]
    if dcs_iter.shape[0] >= 2:
        phases.append(("later", dcs_iter[1:].mean(axis=0)))
    for phase, pred in phases:
        ql = quantile_loss(y, pred, tau)
        rows.append(
            {
                "rep": rep,
                "phase": phase,
                "R": R,
                "tau": tau,
                "ql_censored": ql,
                "ql_benchmark": ql_benchmark,
                "ql_centralized": ql_central,
                "ql_ratio": ql / ql_benchmark if ql_benchmark > 0 else np.nan,
                "ree": ql / ql_central if ql_central > 0 else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--R-grid", type=int, nargs="*", default=list(ILEA_ROUNDS_GRID))
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 20 for 25%% censoring and 50 for 50%%.")
    parser.add_argument("--tau", type=float, default=0.5)
    args = apply_quick_defaults(parser.parse_args(), DIST_N_TRAIN, DIST_N_TEST, DIST_N_REP)

    run_dir = make_run_dir(args.out, "5_5_4_ilea_rounds")
    metric_rows = []
    timing_rows = []
    fit_rows = []
    iteration_rows = []
    J_map, lam_map = load_hyperparameter_map(args.hyperparams, args.J, args.lam, 1, "normal", "right", 0.25)
    total = args.n_rep * len(args.R_grid)
    with tqdm(total=total, desc="ILEA sensitivity") as pbar:
        for R in args.R_grid:
            setting_rows = []
            for rep in range(1, args.n_rep + 1):
                # Use common random numbers across different R values. This
                # keeps the training/test data, benchmark fit, centralized DA
                # fit, and worker partition fixed for the same replication, so
                # the ILEA sensitivity table isolates the effect of R.
                rows, fits, times = run_distributed_replication(
                    1,
                    "normal",
                    "right",
                    0.25,
                    args.n_train,
                    args.n_test,
                    args.K,
                    R,
                    rep,
                    args.test_seed,
                    args.seed,
                    J_map,
                    lam_map,
                    args.max_iter,
                    args.bootstrap_reps,
                    run_dir / "raw",
                    S=args.S,
                    include_centralized=True,
                    include_one_shot=False,
                    target_taus=(args.tau,),
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
                metric_rows.extend(rows)
                setting_rows.extend(rows)
                fit_rows.extend(fits)
                timing_rows.extend(times)
                raw_path = run_dir / "raw" / f"rep_{rep:04d}_s1_normal_right_25_K{args.K}_R{R}.npz"
                iteration_rows.extend(_ilea_iteration_rows(raw_path, rep, R, args.tau))
                pbar.set_postfix(rep=rep, R=R)
                pbar.update(1)
            print_setting_summary(
                setting_rows,
                f"5.5.4 K={args.K}, R={R}, tau={args.tau}",
                ["method", "tau"],
                ["ql_ratio", "ree"],
            )
            save_progress(
                run_dir,
                metric_rows,
                fit_rows,
                timing_rows,
                metric_group_cols=["method", "K", "R", "tau"],
                timing_group_cols=["method", "K", "R"],
            )
            raw_iter = save_csv(run_dir / "iteration_metrics_raw.csv", iteration_rows)
            save_csv(run_dir / "summary_iteration.csv", summarize(raw_iter, ["phase", "R", "tau"], ["ql_ratio", "ree"]))
            save_csv(
                run_dir / "paper_table_ilea_rounds.csv",
                raw_iter.groupby(["phase", "R", "tau"], dropna=False)[["ql_ratio", "ree"]].mean().reset_index(),
            )

    save_json(run_dir / "run_config.json", vars(args))
    print(f"saved ILEA-round sensitivity outputs to {run_dir}")


if __name__ == "__main__":
    main()
