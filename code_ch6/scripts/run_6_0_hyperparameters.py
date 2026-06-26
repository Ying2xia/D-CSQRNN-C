#!/usr/bin/env python3
"""Run Chapter 6 EBIC hyperparameter selection for real-data experiments."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import ROOT, print_runtime_device
from common.config import J_GRID, LAMBDA_GRID
from common.real_config import (
    CENSOR_RATES,
    CENSOR_TYPES,
    CENTRAL_CENSOR_RATE,
    CENTRAL_DATASETS,
    CENTRAL_TRAIN_FRACTION,
    HOUSEHOLD_N_TRAIN,
    TARGET_TAUS,
)
from common.real_data import apply_real_censoring, load_real_dataset, split_and_standardize
from common.storage import make_run_dir, save_csv, save_json
from common.training import select_hyperparameters


ALL_DATASETS = (*CENTRAL_DATASETS, "household_power")


def _dataset_seed(base_seed: int, dataset: str) -> int:
    return int(base_seed + {"boston": 101, "gilgais": 202, "household_power": 303}[dataset])


def _rates_for_dataset(dataset: str, requested: list[float] | None) -> list[float]:
    if requested:
        return [float(x) for x in requested]
    if dataset in CENTRAL_DATASETS:
        return [float(CENTRAL_CENSOR_RATE)]
    return [float(x) for x in CENSOR_RATES]


def _select_one(
    X,
    y,
    tau: float,
    rng: np.random.Generator,
    args,
    J_grid,
    lambda_grid,
    loss_kind: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    return select_hyperparameters(
        X,
        y,
        tau,
        rng,
        J_grid=J_grid,
        lambda_grid=lambda_grid,
        max_iter=args.max_iter,
        loss_kind=loss_kind,
        epsilon=args.epsilon_huber,
        df_mode=args.df_mode,
        active_threshold=args.active_threshold,
        n_restarts=args.n_restarts,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )


def _selection_source(train, min_uncensored: int = 20):
    mask = train.uncensored_mask
    if int(mask.sum()) >= int(min_uncensored):
        return train.X[mask], train.y_obs[mask]
    return train.X, train.y_obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--data-dir", default=str(ROOT), help="Directory containing household_power_consumption.txt and data cache.")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=list(ALL_DATASETS), choices=list(ALL_DATASETS))
    parser.add_argument("--scope", choices=["all", "benchmark", "censored"], default="all")
    parser.add_argument("--censor-types", nargs="*", default=list(CENSOR_TYPES), choices=list(CENSOR_TYPES))
    parser.add_argument("--censor-rates", type=float, nargs="*", default=None)
    parser.add_argument("--household-hyper-n", type=int, default=20_000, help="Household subset size used only for EBIC selection.")
    parser.add_argument("--household-max-rows", type=int, default=None, help="Optional household row cap while loading data.")
    parser.add_argument("--central-train-fraction", type=float, default=CENTRAL_TRAIN_FRACTION)
    parser.add_argument("--J-grid", type=int, nargs="*", default=None)
    parser.add_argument("--lambda-grid", type=float, nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument(
        "--df-mode",
        choices=["selected_variables", "active_hidden_nodes", "hidden_weights", "parameters"],
        default="selected_variables",
        help="EBIC df. Default follows Hao et al.: number of selected input variables.",
    )
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--epsilon-huber", type=float, default=0.1)
    parser.add_argument("--include-daqrnn", dest="include_daqrnn", action="store_true", default=True)
    parser.add_argument("--skip-daqrnn", dest="include_daqrnn", action="store_false")
    parser.add_argument("--max-settings", type=int, default=None, help="Stop after this many tau-level selections.")
    parser.add_argument("--backend", choices=["numpy", "torch", "auto"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    if args.quick:
        J_grid = args.J_grid or [1, 2]
        lambda_grid = args.lambda_grid or [0.01, 0.05]
        args.max_iter = args.max_iter or 25
        args.n_restarts = args.n_restarts or 1
        args.household_hyper_n = min(args.household_hyper_n, 1000)
        args.datasets = args.datasets[:1]
        args.censor_types = args.censor_types[:1]
        args.max_settings = args.max_settings or 4
    else:
        J_grid = args.J_grid or list(J_GRID)
        lambda_grid = args.lambda_grid or list(LAMBDA_GRID)
        args.max_iter = args.max_iter or 120
        args.n_restarts = args.n_restarts or 3

    print_runtime_device(args)
    run_dir = make_run_dir(args.out, "6_0_hyperparameters")
    selected_rows: list[dict[str, object]] = []
    grid_frames: list[pd.DataFrame] = []
    done = 0

    for dataset_name in tqdm(args.datasets, desc="Datasets"):
        data = load_real_dataset(
            dataset_name,
            args.data_dir,
            max_rows=args.household_max_rows if dataset_name == "household_power" else None,
        )
        split_rng = np.random.default_rng(_dataset_seed(args.seed, dataset_name))
        if dataset_name == "household_power":
            n_train = min(int(args.household_hyper_n), data.n - 1)
            split = split_and_standardize(data, split_rng, n_train=n_train, n_test=1)
        else:
            split = split_and_standardize(data, split_rng, train_fraction=args.central_train_fraction)

        if args.scope in {"all", "benchmark"}:
            benchmark_methods = [("CS-QRNN-star", "cs")]
            if args.include_daqrnn and dataset_name in CENTRAL_DATASETS:
                benchmark_methods.append(("DAqrnn-star", "huber"))
            for method, loss_kind in benchmark_methods:
                for tau in tqdm(TARGET_TAUS, desc=f"{dataset_name}/{method}", leave=False):
                    best, grid = _select_one(
                        split.X_train,
                        split.y_train,
                        tau,
                        split_rng,
                        args,
                        J_grid,
                        lambda_grid,
                        loss_kind,
                    )
                    meta = {
                        "dataset": dataset_name,
                        "method": method,
                        "tau": tau,
                        "selection_n": int(len(split.y_train)),
                        "loss_kind": loss_kind,
                    }
                    selected_rows.append({**meta, **best})
                    grid_frames.append(grid.assign(**meta))
                    done += 1
                    if args.max_settings and done >= args.max_settings:
                        break
                if args.max_settings and done >= args.max_settings:
                    break

        if args.max_settings and done >= args.max_settings:
            break

        if args.scope in {"all", "censored"}:
            methods = [("DCS-QRNN-C" if dataset_name == "household_power" else "DA-CSQRNN", "cs")]
            if args.include_daqrnn and dataset_name in CENTRAL_DATASETS:
                methods.append(("DAqrnn", "huber"))
            for censor_rate in _rates_for_dataset(dataset_name, args.censor_rates):
                for censor_type in args.censor_types:
                    censor_rng = np.random.default_rng(_dataset_seed(args.seed, dataset_name) + int(1000 * censor_rate) + CENSOR_TYPES.index(censor_type))
                    train = apply_real_censoring(
                        split.X_train,
                        split.y_train,
                        dataset_name,
                        censor_type,
                        censor_rate,
                        censor_rng,
                    )
                    X_sel, y_sel = _selection_source(train)
                    for method, loss_kind in methods:
                        for tau in tqdm(TARGET_TAUS, desc=f"{dataset_name}/{censor_type}/{method}", leave=False):
                            best, grid = _select_one(
                                X_sel,
                                y_sel,
                                tau,
                                split_rng,
                                args,
                                J_grid,
                                lambda_grid,
                                loss_kind,
                            )
                            meta = {
                                "dataset": dataset_name,
                                "method": method,
                                "censor_type": censor_type,
                                "censor_rate": float(censor_rate),
                                "actual_censor_rate": float((train.delta != 0).mean()),
                                "tau": tau,
                                "selection_n": int(len(y_sel)),
                                "loss_kind": loss_kind,
                            }
                            selected_rows.append({**meta, **best})
                            grid_frames.append(grid.assign(**meta))
                            done += 1
                            if args.max_settings and done >= args.max_settings:
                                break
                        if args.max_settings and done >= args.max_settings:
                            break
                    if args.max_settings and done >= args.max_settings:
                        break
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
            **vars(args),
            "J_grid": list(J_grid),
            "lambda_grid": list(lambda_grid),
            "n_selected_rows": int(len(selected)),
        },
    )
    print(f"saved {len(selected)} selected rows to {run_dir}")


if __name__ == "__main__":
    main()
