#!/usr/bin/env python3
"""Run Section 5.5.5: sensitivity to the number of workers."""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults
from common.augmentation import run_da_csqrnn
from common.config import DIST_N_REP, DIST_N_TEST, DIST_N_TRAIN, WORKER_SENSITIVITY_K, distributed_da_iterations
from common.data import make_censored_dataset, make_tau_grid, make_uncensored_dataset, true_conditional_quantiles
from common.distributed import partition_dataset, run_dcs_qrnn_c
from common.distributed_reporting import print_setting_summary, save_progress, save_worker_paper_table
from common.experiments import benchmark_predictions, load_hyperparameter_map
from common.metrics import metric_rows_from_predictions, quantile_loss
from common.storage import make_run_dir, save_json, save_npz
from common.training import tau_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--K-grid", type=int, nargs="*", default=list(WORKER_SENSITIVITY_K))
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 20 for 25%% censoring and 50 for 50%%.")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()
    # Worker-sensitivity timing should be comparable across K. Force the CPU
    # NumPy path so RACT is not affected by GPU batching or single-GPU
    # contention, and keep R=1 as the default ILEA correction setting.
    args.backend = "numpy"
    args.device = "cpu"
    args = apply_quick_defaults(args, DIST_N_TRAIN, DIST_N_TEST, DIST_N_REP)

    run_dir = make_run_dir(args.out, "5_5_5_worker_sensitivity")
    metric_rows = []
    timing_rows = []
    fit_rows = []
    J_map, lam_map = load_hyperparameter_map(args.hyperparams, args.J, args.lam, 1, "normal", "right", 0.25)
    total = args.n_rep * len(args.K_grid)
    target_taus = [tau_key(args.tau)]
    S = int(distributed_da_iterations(0.25) if args.S is None else args.S)
    bench_S = max(int(args.bootstrap_reps or 0), S)
    setting_rows_by_K = {int(K): [] for K in args.K_grid}
    setting_times_by_K = {int(K): [] for K in args.K_grid}
    with tqdm(total=total, desc="Worker sensitivity") as pbar:
        for rep in range(1, args.n_rep + 1):
            rng = np.random.default_rng(args.seed + rep)
            test_rng = np.random.default_rng(args.test_seed)
            train = make_censored_dataset(args.n_train, 1, "normal", "right", 0.25, rng)
            test = make_uncensored_dataset(args.n_test, 1, "normal", test_rng)
            tau_grid = make_tau_grid(args.n_train)
            true_q = true_conditional_quantiles(test.X, 1, "normal", target_taus)
            actual_censor_rate = float((train.delta != 0).mean())

            benchmark_pred, benchmark_boot, bench_lower, bench_upper, _ = benchmark_predictions(
                train,
                test.X,
                target_taus,
                J_map,
                lam_map,
                rng,
                bootstrap_reps=bench_S,
                max_iter=args.max_iter,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )

            # The centralized DA reference is independent of K, so compute it
            # once per Monte Carlo replication and reuse it for every worker
            # count. REE and RACT still use this reference as their denominator.
            t0 = time.perf_counter()
            central_da = run_da_csqrnn(
                train,
                test.X,
                target_taus,
                J_map,
                lam_map,
                S,
                tau_grid,
                rng,
                max_iter=args.max_iter,
                backend=args.backend,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
            central_time = time.perf_counter() - t0
            central_state = copy.deepcopy(rng.bit_generator.state)
            central_ql = {
                tau: quantile_loss(test.y_true, central_da.final_predictions[:, j], tau)
                for j, tau in enumerate(target_taus)
            }

            for K in args.K_grid:
                K = int(K)
                rng_k = np.random.default_rng()
                rng_k.bit_generator.state = copy.deepcopy(central_state)
                partitions = partition_dataset(train, K, rng_k)
                dcs = run_dcs_qrnn_c(
                    partitions,
                    test.X,
                    target_taus,
                    J_map,
                    lam_map,
                    S,
                    args.R,
                    tau_grid,
                    rng_k,
                    max_iter=args.max_iter,
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
                meta = {
                    "rep": rep,
                    "scenario": 1,
                    "error": "normal",
                    "censor_type": "right",
                    "censor_rate": 0.25,
                    "method": "DCS-QRNN-C",
                    "K": K,
                    "R": args.R,
                    "n_train": args.n_train,
                    "n_test": args.n_test,
                    "S": S,
                    "benchmark_reps": bench_S,
                    "actual_censor_rate": actual_censor_rate,
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
                for row in dcs_rows:
                    ql_central = central_ql[row["tau"]]
                    row["ql_centralized"] = ql_central
                    row["ree"] = row["ql_censored"] / ql_central if ql_central > 0 else np.nan
                dcs_time = {
                    **meta,
                    "time_seconds": dcs.elapsed_seconds,
                    "ract": central_time / dcs.elapsed_seconds if dcs.elapsed_seconds > 0 else np.nan,
                }
                dcs_fits = [
                    {
                        **row,
                        **{k: meta[k] for k in ("rep", "scenario", "error", "censor_type", "censor_rate", "method", "K", "R")},
                    }
                    for row in dcs.fit_rows
                ]

                save_npz(
                    run_dir / "raw" / f"rep_{rep:04d}_s1_normal_right_25_K{K}_R{args.R}.npz",
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
                    pred_central=central_da.final_predictions,
                    pred_central_iter=central_da.iteration_predictions,
                    lower_central=central_da.lower,
                    upper_central=central_da.upper,
                    pred_dcs=dcs.final_predictions,
                    pred_dcs_iter=dcs.iteration_predictions,
                    lower_dcs=dcs.lower,
                    upper_dcs=dcs.upper,
                    pred_one_shot=np.array([]),
                    tau_schedule_dcs=dcs.tau_schedule,
                )

                metric_rows.extend(dcs_rows)
                setting_rows_by_K[K].extend(dcs_rows)
                fit_rows.extend(dcs_fits)
                timing_rows.append(dcs_time)
                setting_times_by_K[K].append(dcs_time)
                pbar.set_postfix(rep=rep, K=K)
                pbar.update(1)
            save_progress(
                run_dir,
                metric_rows,
                fit_rows,
                timing_rows,
                metric_group_cols=["method", "K", "R", "tau"],
                metric_summary_name="summary_accuracy.csv",
                timing_group_cols=["method", "K", "R"],
                timing_summary_name="summary_timing.csv",
            )
            save_worker_paper_table(run_dir, metric_rows, timing_rows, args.n_train, args.tau)

        for K in args.K_grid:
            K = int(K)
            print_setting_summary(
                setting_rows_by_K[K],
                f"5.5.5 K={K}, R={args.R}, tau={args.tau}",
                ["method", "tau"],
                ["ql_ratio", "ree"],
            )
            print_setting_summary(
                setting_times_by_K[K],
                f"5.5.5 timing K={K}, R={args.R}",
                ["method", "K", "R"],
                ["time_seconds", "ract"],
            )

    save_json(run_dir / "run_config.json", vars(args))
    print(f"saved worker-sensitivity outputs to {run_dir}")


if __name__ == "__main__":
    main()
