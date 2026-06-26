#!/usr/bin/env python3
"""Run Section 5.4.1: EBIC hyperparameter selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from _common import ROOT, add_filter_args, print_runtime_device
from common.config import CENTRAL_N_TRAIN, ERRORS, J_GRID, LAMBDA_GRID, SCENARIOS, TARGET_TAUS, central_settings
from common.data import make_censored_dataset, make_uncensored_dataset
from common.storage import make_run_dir, save_csv, save_json
from common.training import select_hyperparameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scope", choices=["all", "benchmark", "censored"], default="all")
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument(
        "--df-mode",
        choices=["selected_variables", "active_hidden_nodes", "hidden_weights", "parameters"],
        default="selected_variables",
        help="EBIC df. Default follows Hao et al.: number of selected input variables.",
    )
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--backend", choices=["numpy", "torch", "auto"], default="auto")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, mps, or auto.")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    add_filter_args(parser)
    args = parser.parse_args()
    print_runtime_device(args)

    import numpy as np

    rng = np.random.default_rng(args.seed)
    run_dir = make_run_dir(args.out, "5_4_1_hyperparameters")
    if args.quick:
        J_grid = J_GRID
        lambda_grid = (0.01, 0.05)
        max_iter = 25
        n_train = 120
        n_restarts = args.n_restarts or 1
    else:
        J_grid = J_GRID
        lambda_grid = LAMBDA_GRID
        max_iter = 120
        n_train = CENTRAL_N_TRAIN
        n_restarts = args.n_restarts or 3

    selected_rows = []
    grid_frames = []
    done = 0

    if args.scope in {"all", "benchmark"}:
        benchmark_combos = [
            (s, e) for s in SCENARIOS for e in ERRORS
            if (args.scenario is None or s == args.scenario)
            and (args.error is None or e == args.error)
        ]
        for scenario, error in tqdm(benchmark_combos, desc="Benchmark EBIC"):
            data = make_uncensored_dataset(n_train, scenario, error, rng)
            for tau in tqdm(TARGET_TAUS, desc=f"S{scenario}/{error} taus", leave=False):
                best, grid = select_hyperparameters(
                    data.X,
                    data.y_true,
                    tau,
                    rng,
                    J_grid=J_grid,
                    lambda_grid=lambda_grid,
                    max_iter=max_iter,
                    df_mode=args.df_mode,
                    active_threshold=args.active_threshold,
                    n_restarts=n_restarts,
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
                meta = {"method": "CS-QRNN-star", "scenario": scenario, "error": error, "tau": tau}
                selected_rows.append({**meta, **best})
                grid_frames.append(grid.assign(**meta))
                done += 1
                if args.max_settings and done >= args.max_settings:
                    break
            if args.max_settings and done >= args.max_settings:
                break

    if args.scope in {"all", "censored"} and not (args.max_settings and done >= args.max_settings):
        censored_settings = list(central_settings())
        censored_settings = [
            (s, e, ct, cr) for s, e, ct, cr in censored_settings
            if (args.scenario is None or s == args.scenario)
            and (args.error is None or e == args.error)
            and (args.censor_type is None or ct == args.censor_type)
            and (args.censor_rate is None or float(cr) == float(args.censor_rate))
        ]
        for scenario, error, censor_type, censor_rate in tqdm(censored_settings, desc="Censored EBIC"):
            data = make_censored_dataset(n_train, scenario, error, censor_type, censor_rate, rng)
            X_sel = data.X[data.uncensored_mask]
            y_sel = data.y_obs[data.uncensored_mask]
            for tau in tqdm(TARGET_TAUS, desc=f"S{scenario}/{error}/{censor_type} taus", leave=False):
                best, grid = select_hyperparameters(
                    X_sel,
                    y_sel,
                    tau,
                    rng,
                    J_grid=J_grid,
                    lambda_grid=lambda_grid,
                    max_iter=max_iter,
                    df_mode=args.df_mode,
                    active_threshold=args.active_threshold,
                    n_restarts=n_restarts,
                    backend=args.backend,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )
                meta = {
                    "method": "DA-CSQRNN",
                    "scenario": scenario,
                    "error": error,
                    "censor_type": censor_type,
                    "censor_rate": censor_rate,
                    "tau": tau,
                    "actual_censor_rate": float((data.delta != 0).mean()),
                }
                selected_rows.append({**meta, **best})
                grid_frames.append(grid.assign(**meta))
                done += 1
                if args.max_settings and done >= args.max_settings:
                    break
            if args.max_settings and done >= args.max_settings:
                break

    selected = save_csv(run_dir / "selected_hyperparameters.csv", selected_rows)
    grid_all = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()
    save_csv(run_dir / "ebic_grid_raw.csv", grid_all)
    save_json(
        run_dir / "run_config.json",
        {
            "quick": args.quick,
            "n_train": n_train,
            "J_grid": list(J_grid),
            "lambda_grid": list(lambda_grid),
            "max_iter": max_iter,
            "df_mode": args.df_mode,
            "active_threshold": args.active_threshold,
            "n_restarts": n_restarts,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "n_selected_rows": int(len(selected)),
        },
    )
    print(f"saved {len(selected)} selected rows to {run_dir}")


if __name__ == "__main__":
    main()
