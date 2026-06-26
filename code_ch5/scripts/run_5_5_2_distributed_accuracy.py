#!/usr/bin/env python3
"""Run Section 5.5.2: distributed statistical accuracy and REE."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from _common import add_common_args, apply_quick_defaults
from common.config import (
    CENSOR_RATES,
    CENSOR_TYPES,
    DIST_N_REP,
    DIST_N_TEST,
    DIST_N_TRAIN,
    DIST_WORKERS,
    ERRORS,
    SCENARIOS,
)
from common.distributed_reporting import (
    print_setting_summary,
    print_timing_summary,
    save_accuracy_paper_tables,
    save_progress,
)
from common.distributed_experiments import run_distributed_replication
from common.experiments import load_hyperparameter_map
from common.storage import make_run_dir, save_json


def _setting_seed(base_seed: int, scenario: int, error: str, censor_type: str, censor_rate: float) -> int:
    error_idx = list(ERRORS).index(error)
    censor_idx = list(CENSOR_TYPES).index(censor_type)
    return int(base_seed + 100_000 * scenario + 10_000 * error_idx + 1_000 * censor_idx + round(100 * censor_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--K", type=int, nargs="*", default=list(DIST_WORKERS))
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--scenarios", type=int, nargs="*", default=list(SCENARIOS))
    parser.add_argument("--errors", nargs="*", default=list(ERRORS), choices=list(ERRORS))
    parser.add_argument("--censor-types", nargs="*", choices=list(CENSOR_TYPES), default=list(CENSOR_TYPES))
    parser.add_argument("--censor-rates", type=float, nargs="*", choices=list(CENSOR_RATES), default=list(CENSOR_RATES))
    parser.add_argument("--scenario", type=int, default=None, choices=list(SCENARIOS), help="Legacy shortcut for one scenario.")
    parser.add_argument("--error", default=None, choices=list(ERRORS), help="Legacy shortcut for one error distribution.")
    parser.add_argument("--censor-type", default=None, choices=list(CENSOR_TYPES), help="Legacy shortcut for one censoring type.")
    parser.add_argument("--censor-rate", type=float, default=None, choices=list(CENSOR_RATES), help="Legacy shortcut for one censoring rate.")
    parser.add_argument("--S", type=int, default=None, help="Override DA iterations; default is 20 for 25%% censoring and 50 for 50%%.")
    parser.add_argument("--include-one-shot", dest="include_one_shot", action="store_true", default=True)
    parser.add_argument("--skip-one-shot", dest="include_one_shot", action="store_false")
    parser.add_argument(
        "--only-one-shot",
        action="store_true",
        help="Run DA-CSQRNN and OS rows, skipping DCS-QRNN-C. The uncensored benchmark is still run for ql_ratio.",
    )
    args = apply_quick_defaults(parser.parse_args(), DIST_N_TRAIN, DIST_N_TEST, DIST_N_REP)
    if args.only_one_shot:
        args.include_one_shot = True
    if args.scenario is not None:
        args.scenarios = [args.scenario]
    if args.error is not None:
        args.errors = [args.error]
    if args.censor_type is not None:
        args.censor_types = [args.censor_type]
    if args.censor_rate is not None:
        args.censor_rates = [args.censor_rate]

    run_dir = make_run_dir(args.out, "5_5_2_distributed_accuracy")
    metric_rows = []
    fit_rows = []
    timing_rows = []
    settings = [
        (scenario, error, censor_type, censor_rate)
        for scenario in args.scenarios
        for error in args.errors
        for censor_type in args.censor_types
        for censor_rate in args.censor_rates
    ]
    first_K = args.K[0] if args.K else None
    with tqdm(total=len(settings), desc="5.5.2 settings") as pbar:
        for scenario, error, censor_type, censor_rate in settings:
            print(f"\nRunning 5.5.2 setting: scenario={scenario}, error={error}, censor={censor_type}, rate={censor_rate:.0%}")
            J_map, lam_map = load_hyperparameter_map(
                args.hyperparams, args.J, args.lam, scenario, error, censor_type, censor_rate
            )
            base_seed = _setting_seed(args.seed, scenario, error, censor_type, censor_rate)
            for K in args.K:
                setting_rows = []
                setting_timing_rows = []
                for rep in range(1, args.n_rep + 1):
                    rows, fits, times = run_distributed_replication(
                        scenario,
                        error,
                        censor_type,
                        censor_rate,
                        args.n_train,
                        args.n_test,
                        K,
                        args.R,
                        rep,
                        args.test_seed,
                        base_seed,
                        J_map,
                        lam_map,
                        args.max_iter,
                        args.bootstrap_reps,
                        run_dir / "raw",
                        S=args.S,
                        include_centralized=True,
                        include_dcs=not args.only_one_shot,
                        include_one_shot=args.include_one_shot,
                        backend=args.backend,
                        device=args.device,
                        torch_dtype=args.torch_dtype,
                    )
                    rows_to_store = rows if K == first_K else [r for r in rows if r.get("method") != "DA-CSQRNN"]
                    times_to_store = times if K == first_K else [r for r in times if r.get("method") != "DA-CSQRNN"]
                    metric_rows.extend(rows_to_store)
                    setting_rows.extend(rows)
                    setting_timing_rows.extend(times)
                    fit_rows.extend(fits if K == first_K else [r for r in fits if r.get("method") != "DA-CSQRNN"])
                    timing_rows.extend(times_to_store)
                print_setting_summary(
                    setting_rows,
                    f"5.5.2 S{scenario}/{error}/{censor_type}/{censor_rate:.0%}, K={K}, R={args.R}",
                    ["method", "tau"],
                    ["ql_ratio", "ree"],
                )
                print_timing_summary(
                    setting_timing_rows,
                    f"5.5.2 timing S{scenario}/{error}/{censor_type}/{censor_rate:.0%}, K={K}, R={args.R}",
                )
                save_progress(
                    run_dir,
                    metric_rows,
                    fit_rows,
                    timing_rows,
                    metric_group_cols=["scenario", "error", "censor_type", "censor_rate", "method", "K", "R", "tau"],
                    timing_group_cols=["scenario", "error", "censor_type", "censor_rate", "method", "K", "R"],
                )
                save_accuracy_paper_tables(run_dir, metric_rows)
            pbar.update(1)

    save_json(run_dir / "run_config.json", vars(args) | {"settings": [list(s) for s in settings]})
    print(f"saved distributed accuracy outputs to {run_dir}")


if __name__ == "__main__":
    main()
