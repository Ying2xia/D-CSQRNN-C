"""Shared experiment orchestration for the section scripts.

Three changes from the original version:

1. ``benchmark_predictions`` runs an *iterative* warm-started bootstrap by
   default, so its prediction-interval samples have the same optimisation-noise
   structure as DA-CSQRNN's S iterations. The ``pi_method="independent"``
   path keeps the original cold-start behaviour available.

2. ``run_comparison_replication`` runs DA-CSQRNN, DAqrnn, and (optionally)
   DeepQuantreg against a benchmark and returns metric rows carrying both
   QL_ratio AND PIW_ratio for every method. This is what Section 5.4.3 needs
   in order to populate a Hao-et-al-style comparison table for the prediction
   interval width.

3. Section 5.4.3 uses method-specific fully observed benchmarks where
   available: CS-QRNN* for DA-CSQRNN, Huber/DAqrnn* for DAqrnn, and
   DeepQuantreg* for DeepQuantreg when the external baseline is enabled.
   This lets ``benchmark_coverage`` explain whether a large PIW_ratio comes
   from the censored method or from a narrow benchmark denominator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .augmentation import run_da_csqrnn
from .baselines import run_daqrnn_baseline
from .config import TARGET_TAUS, da_iterations
from .data import (
    SimDataset,
    as_fully_observed,
    make_censored_dataset,
    make_tau_grid,
    make_uncensored_dataset,
    true_conditional_quantiles,
)
from .metrics import interval_bounds, metric_rows_from_predictions, quantile_loss, piw
from .storage import save_npz
from .training import bootstrap_predictions, fit_models_for_taus, tau_key


def load_hyperparameter_map(
    csv_path: str | Path | None,
    default_J: int,
    default_lambda: float,
    scenario: int | None = None,
    error: str | None = None,
    censor_type: str | None = None,
    censor_rate: float | None = None,
    method: str | None = None,
) -> tuple[dict[float, int], dict[float, float]]:
    """Load per-tau hyperparameters when a Section 5.4.1 CSV is available."""

    J_map = {tau_key(t): int(default_J) for t in TARGET_TAUS}
    lam_map = {tau_key(t): float(default_lambda) for t in TARGET_TAUS}
    if not csv_path:
        return J_map, lam_map
    path = Path(csv_path)
    if not path.exists():
        return J_map, lam_map
    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    filters = {
        "scenario": scenario,
        "error": error,
        "censor_type": censor_type,
        "censor_rate": censor_rate,
        "method": method,
    }
    for col, value in filters.items():
        if value is not None and col in df.columns:
            mask &= df[col] == value
    sub = df[mask]
    for _, row in sub.iterrows():
        tau = tau_key(row["tau"])
        if "J" in row:
            J_map[tau] = int(row["J"])
        if "lambda" in row:
            lam_map[tau] = float(row["lambda"])
    return J_map, lam_map


def _iterative_bootstrap_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    target_taus: list[float],
    J,
    lam,
    rng: np.random.Generator,
    S: int,
    max_iter: int,
    backend: str,
    device: str,
    torch_dtype: str,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
):
    """S iterations of (paired bootstrap + warm-start fit) on uncensored data.

    Mirrors DA-CSQRNN's loop without the imputation step. The first iteration
    fits from a random init; every subsequent iteration warm-starts from the
    previous fit, so the variability between iterations reflects data
    resampling rather than optimiser jitter.
    """

    target_taus = [tau_key(t) for t in target_taus]
    n = len(y_train)
    rows: list[dict[str, object]] = []

    base_models, base_rows = fit_models_for_taus(
        X_train, y_train, target_taus, J, lam, rng,
        max_iter=max_iter, loss_kind=loss_kind, epsilon=epsilon,
        backend=backend, device=device, torch_dtype=torch_dtype,
    )
    pred = np.column_stack([base_models[tau].predict(X_test) for tau in target_taus])
    for r in base_rows:
        r = dict(r)
        r["benchmark_iteration"] = 0
        r["stage"] = "benchmark_initial"
        rows.append(r)

    iter_preds = np.empty((int(S), X_test.shape[0], len(target_taus)), dtype=float)
    models = dict(base_models)
    for s in range(int(S)):
        idx = rng.integers(0, n, size=n)
        updated, fit_rows = fit_models_for_taus(
            X_train[idx], y_train[idx], target_taus, J, lam, rng,
            max_iter=max_iter,
            warm_starts={t: models[t] for t in target_taus if t in models},
            loss_kind=loss_kind,
            epsilon=epsilon,
            backend=backend, device=device, torch_dtype=torch_dtype,
        )
        models.update(updated)
        for j, tau in enumerate(target_taus):
            iter_preds[s, :, j] = models[tau].predict(X_test)
        for r in fit_rows:
            r = dict(r)
            r["benchmark_iteration"] = s + 1
            r["stage"] = "benchmark_update"
            rows.append(r)
    lower, upper = interval_bounds(iter_preds)
    return pred, iter_preds, lower, upper, rows


def benchmark_predictions(
    train: SimDataset,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J,
    lam,
    rng: np.random.Generator,
    bootstrap_reps: int,
    max_iter: int,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
    pi_method: str = "iterative",
    loss_kind: str = "cs",
    epsilon: float = 0.1,
):
    """Benchmark (uncensored) predictions and prediction interval.

    Parameters
    ----------
    train
        Simulated training data. The benchmark intentionally uses
        ``train.y_true`` so it represents the fully observed CS-QRNN* model.
    X_test
        Test predictors where benchmark quantiles are predicted.
    target_taus
        Quantile levels to fit and evaluate.
    J, lam
        Hidden-node count and regularization strength for the benchmark
        models. Each may be scalar or a per-tau map.
    rng
        Random generator used for bootstrap samples and model initialization.
    bootstrap_reps
        Number of prediction samples used to build the benchmark PI.
    max_iter
        Adam optimization steps for each QRNN fit.
    backend, device, torch_dtype
        Compute backend controls passed through to model fitting.
    pi_method : {"iterative", "independent"}
        How prediction-interval bootstrap samples are generated. The default
        ``"iterative"`` runs S = ``bootstrap_reps`` warm-started bootstrap
        iterations and is what the in-process DA methods expect.
        ``"independent"`` runs ``bootstrap_reps`` cold-start fits; this is the
        regime that matches an out-of-process baseline like DeepQuantreg.
    loss_kind, epsilon
        Loss used by the fully observed benchmark. DA-CSQRNN uses
        ``loss_kind="cs"``; DAqrnn uses ``loss_kind="huber"`` with the same
        ``epsilon`` as its censored fit.
    """

    target_taus = [tau_key(t) for t in target_taus]
    if pi_method == "iterative":
        return _iterative_bootstrap_predictions(
            train.X, train.y_true, X_test, target_taus, J, lam, rng,
            S=int(bootstrap_reps), max_iter=max_iter,
            backend=backend, device=device, torch_dtype=torch_dtype,
            loss_kind=loss_kind, epsilon=epsilon,
        )
    if pi_method != "independent":
        raise ValueError(f"unknown pi_method: {pi_method}")

    models, rows = fit_models_for_taus(
        train.X,
        train.y_true,
        target_taus,
        J,
        lam,
        rng,
        max_iter=max_iter,
        loss_kind=loss_kind,
        epsilon=epsilon,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    pred = np.column_stack([models[tau].predict(X_test) for tau in target_taus])
    boot, boot_rows = bootstrap_predictions(
        train.X,
        train.y_true,
        X_test,
        target_taus,
        J,
        lam,
        rng,
        B=int(bootstrap_reps),
        max_iter=max_iter,
        loss_kind=loss_kind,
        epsilon=epsilon,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    lower, upper = interval_bounds(boot)
    return pred, boot, lower, upper, rows + boot_rows


def run_centralized_replication(
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    n_train: int,
    n_test: int,
    rep: int,
    test_seed: int,
    base_seed: int,
    J,
    lam,
    max_iter: int,
    bootstrap_reps: int,
    raw_dir: Path,
    S: int | None = None,
    J_benchmark=None,
    lam_benchmark=None,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
    pi_method: str = "iterative",
):
    """Run one centralized Section 5.4.2 Monte Carlo replicate.

    Parameters
    ----------
    scenario, error, censor_type, censor_rate
        Simulation setting from Chapter 5: data-generating scenario, error
        distribution, censoring mechanism, and nominal censoring rate.
    n_train, n_test
        Training and fixed-test sample sizes.
    rep, test_seed, base_seed
        Replicate index and seeds. The training seed is ``base_seed + rep``;
        the test seed is fixed so methods are compared on the same test set.
    J, lam
        DA-CSQRNN hyperparameters, either scalars or per-tau maps.
    max_iter
        Adam optimization steps for each QRNN fit.
    bootstrap_reps
        Minimum benchmark PI sample size. Use ``0`` to make the benchmark use
        exactly ``S`` bootstrap/iteration predictions. Values larger than
        ``S`` are only for denominator-stability sensitivity checks; they do
        not add extra DA bootstrap steps.
    raw_dir
        Directory where the replicate-level ``.npz`` file is saved.
    S
        Optional DA iteration override. If omitted, ``da_iterations`` supplies
        the current default schedule from ``common.config``.
    J_benchmark, lam_benchmark
        Fully observed CS-QRNN* benchmark hyperparameters. If omitted, the DA
        hyperparameters are reused.
    backend, device, torch_dtype
        Compute backend controls passed through to QRNN fitting.
    pi_method
        Benchmark PI method: ``"iterative"`` for warm-start bootstrap or
        ``"independent"`` for ordinary cold-start bootstrap sensitivity checks.
    """
    rng = np.random.default_rng(base_seed + rep)
    test_rng = np.random.default_rng(test_seed)
    train = make_censored_dataset(n_train, scenario, error, censor_type, censor_rate, rng)
    test = make_uncensored_dataset(n_test, scenario, error, test_rng)
    tau_grid = make_tau_grid(n_train)
    S = int(da_iterations(censor_rate) if S is None else S)
    true_q = true_conditional_quantiles(test.X, scenario, error, TARGET_TAUS)
    if J_benchmark is None:
        J_benchmark = J
    if lam_benchmark is None:
        lam_benchmark = lam

    bench_S = max(int(bootstrap_reps or 0), S)
    benchmark_pred, benchmark_boot, bench_lower, bench_upper, bench_rows = benchmark_predictions(
        train,
        test.X,
        TARGET_TAUS,
        J_benchmark,
        lam_benchmark,
        rng,
        bootstrap_reps=bench_S,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
        pi_method=pi_method,
    )
    da = run_da_csqrnn(
        train,
        test.X,
        TARGET_TAUS,
        J,
        lam,
        S=S,
        tau_grid=tau_grid,
        rng=rng,
        max_iter=max_iter,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    metadata = {
        "rep": rep,
        "scenario": scenario,
        "error": error,
        "censor_type": censor_type,
        "censor_rate": censor_rate,
        "method": "DA-CSQRNN",
        "n_train": n_train,
        "n_test": n_test,
        "S": S,
        "benchmark_reps": bench_S,
        "actual_censor_rate": float((train.delta != 0).mean()),
        "pi_benchmark": pi_method,
        "benchmark_method": "CS-QRNN-star",
    }
    rows = metric_rows_from_predictions(
        test.y_true,
        list(TARGET_TAUS),
        da.final_predictions,
        benchmark_pred,
        da.lower,
        da.upper,
        bench_lower,
        bench_upper,
        metadata,
        true_quantiles=true_q,
    )
    save_npz(
        raw_dir / f"rep_{rep:04d}_s{scenario}_{error}_{censor_type}_{int(censor_rate*100)}.npz",
        X_test=test.X,
        y_test=test.y_true,
        X_train=train.X,
        y_train_true=train.y_true,
        y_train_obs=train.y_obs,
        delta=train.delta,
        L=train.L,
        R=train.R,
        target_taus=np.array(TARGET_TAUS),
        true_quantiles=true_q,
        tau_grid=tau_grid,
        benchmark_reps=np.array(bench_S),
        tau_schedule=da.tau_schedule,
        pred_da=da.final_predictions,
        pred_da_iter=da.iteration_predictions,
        lower_da=da.lower,
        upper_da=da.upper,
        pred_benchmark=benchmark_pred,
        pred_benchmark_boot=benchmark_boot,
        lower_benchmark=bench_lower,
        upper_benchmark=bench_upper,
        last_imputed_y=da.last_imputed_y,
    )
    fit_rows = da.fit_rows + bench_rows
    for row in fit_rows:
        row.update({k: metadata[k] for k in ("rep", "scenario", "error", "censor_type", "censor_rate")})
    return rows, fit_rows


# ----------------------------------------------------------------------------
# Section 5.4.3 comparison: DA-CSQRNN vs DAqrnn (vs DeepQuantreg)
# ----------------------------------------------------------------------------


def run_comparison_replication(
    scenario: int,
    error: str,
    censor_type: str,
    censor_rate: float,
    n_train: int,
    n_test: int,
    rep: int,
    test_seed: int,
    base_seed: int,
    J_dacsqrnn,
    lam_dacsqrnn,
    J_daqrnn,
    lam_daqrnn,
    max_iter: int,
    bootstrap_reps: int,
    raw_dir: Path,
    S: int | None = None,
    J_benchmark=None,
    lam_benchmark=None,
    epsilon_huber: float = 0.1,
    include_deepquantreg: bool = False,
    include_dacsqrnn: bool = True,
    deepquantreg_bootstrap_reps: int | None = None,
    deepquantreg_pi_benchmark: str = "own",
    target_taus: Sequence[float] = TARGET_TAUS,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
    pi_method: str = "iterative",
):
    """One Section 5.4.3 replicate: DA-CSQRNN, DAqrnn, optional DeepQuantreg.

    DA-CSQRNN is compared with a fully observed CS-QRNN* benchmark, while
    DAqrnn is compared with a fully observed Huber/DAqrnn* benchmark.
    DeepQuantreg's prediction interval is built from cold-start bootstrap
    because the external program cannot be warm-started;
    ``deepquantreg_pi_benchmark`` controls which benchmark its PIW_ratio is
    computed against.

    Hyperparameters are accepted per-method because in Hao et al.'s setup the
    EBIC search is run separately for each method. Pass the same value for both
    if your search is shared. ``J_benchmark`` / ``lam_benchmark`` default to
    the DA-CSQRNN hyperparameters, matching how the benchmark is treated in
    the Section 5.4.2 tables.

    Parameters that are new relative to the QL-only version
    -------------------------------------------------------
    include_deepquantreg : bool
        Run DeepQuantreg via ``$DEEPQUANTREG_CMD``. With this off the function
        produces only the two in-process method rows and behaves like the
        previous version.
    include_dacsqrnn : bool
        Run the DA-CSQRNN method inside Section 5.4.3. Set this to ``False``
        when reusing DA-CSQRNN rows already produced by Section 5.4.2.
    deepquantreg_bootstrap_reps : int | None
        Number of cold-start DeepQuantreg fits used to build its prediction
        interval. Defaults to the benchmark PI sample count, which is ``S``
        unless ``bootstrap_reps`` is set larger than ``S``. Set to ``0`` to
        skip the PI bootstrap and emit ``NaN`` for DeepQuantreg's piw_* columns.
    deepquantreg_pi_benchmark : {"own", "shared", "matched"}
        ``"own"``: DeepQuantreg uses a fully observed DeepQuantreg* benchmark.
        This is the fairest coverage/PIW comparison but costs additional
        external DeepQuantreg calls.
        ``"shared"``: DeepQuantreg uses the warm-start CS-QRNN* benchmark in
        its PIW_ratio denominator. Simple, but not method-matched.
        ``"matched"``: an additional cold-start independent benchmark is fit
        and used as DeepQuantreg's denominator only.
    pi_method : {"iterative", "independent"}
        Benchmark PI method used by DA-CSQRNN and DAqrnn. ``"independent"``
        is mainly a denominator sensitivity check.

    Returns
    -------
    metric_rows : list[dict]
        One row per (method, tau). Each row carries ``ql_censored``,
        ``ql_benchmark``, ``ql_ratio``, ``piw_censored``, ``piw_benchmark``,
        ``piw_ratio`` plus the metadata columns.
    fit_rows : list[dict]
        Optimiser diagnostics from every in-process fit (DA-CSQRNN iterations,
        DAqrnn iterations, benchmark iterations).
    """

    rng = np.random.default_rng(base_seed + rep)
    test_rng = np.random.default_rng(test_seed)
    train = make_censored_dataset(n_train, scenario, error, censor_type, censor_rate, rng)
    test = make_uncensored_dataset(n_test, scenario, error, test_rng)
    tau_grid = make_tau_grid(n_train)
    target_taus = [tau_key(t) for t in target_taus]
    S = int(da_iterations(censor_rate) if S is None else S)
    true_q = true_conditional_quantiles(test.X, scenario, error, target_taus)
    bench_S = max(int(bootstrap_reps or 0), S)
    if deepquantreg_bootstrap_reps is None:
        deepquantreg_bootstrap_reps = int(bench_S)
    if deepquantreg_pi_benchmark not in {"own", "shared", "matched"}:
        raise ValueError(f"unknown deepquantreg_pi_benchmark: {deepquantreg_pi_benchmark}")

    if J_benchmark is None:
        J_benchmark = J_dacsqrnn
    if lam_benchmark is None:
        lam_benchmark = lam_dacsqrnn

    # Method-specific fully observed benchmarks. This keeps each PIW ratio and
    # benchmark coverage aligned with the loss used by that method.
    benchmark_pred, benchmark_boot, bench_lower, bench_upper, bench_rows = benchmark_predictions(
        train, test.X, target_taus, J_benchmark, lam_benchmark, rng,
        bootstrap_reps=bench_S, max_iter=max_iter,
        backend=backend, device=device, torch_dtype=torch_dtype,
        pi_method=pi_method, loss_kind="cs",
    )
    benchmark_huber_pred, benchmark_huber_boot, huber_bench_lower, huber_bench_upper, huber_bench_rows = benchmark_predictions(
        train, test.X, target_taus, J_daqrnn, lam_daqrnn, rng,
        bootstrap_reps=bench_S, max_iter=max_iter,
        backend=backend, device=device, torch_dtype=torch_dtype,
        pi_method=pi_method, loss_kind="huber", epsilon=epsilon_huber,
    )
    bench_rows = list(bench_rows) + list(huber_bench_rows)

    da_cs = None
    if include_dacsqrnn:
        da_cs = run_da_csqrnn(
            train, test.X, target_taus, J_dacsqrnn, lam_dacsqrnn,
            S=S, tau_grid=tau_grid, rng=rng,
            max_iter=max_iter, loss_kind="cs",
            backend=backend, device=device, torch_dtype=torch_dtype,
        )

    da_huber = run_daqrnn_baseline(
        train, test.X, target_taus, J_daqrnn, lam_daqrnn,
        S=S, tau_grid=tau_grid, rng=rng,
        max_iter=max_iter, epsilon=epsilon_huber,
        backend=backend, device=device, torch_dtype=torch_dtype,
    )

    base_meta = {
        "rep": rep,
        "scenario": scenario,
        "error": error,
        "censor_type": censor_type,
        "censor_rate": censor_rate,
        "n_train": n_train,
        "n_test": n_test,
        "S": S,
        "benchmark_reps": bench_S,
        "actual_censor_rate": float((train.delta != 0).mean()),
    }

    metric_rows: list[dict[str, object]] = []
    if da_cs is not None:
        metric_rows.extend(metric_rows_from_predictions(
            test.y_true, list(target_taus),
            da_cs.final_predictions, benchmark_pred,
            da_cs.lower, da_cs.upper, bench_lower, bench_upper,
            {
                **base_meta,
                "method": "DA-CSQRNN",
                "pi_benchmark": pi_method,
                "benchmark_method": "CS-QRNN-star",
            },
            true_quantiles=true_q,
        ))
    metric_rows.extend(metric_rows_from_predictions(
        test.y_true, list(target_taus),
        da_huber.final_predictions, benchmark_huber_pred,
        da_huber.lower, da_huber.upper, huber_bench_lower, huber_bench_upper,
        {
            **base_meta,
            "method": "DAqrnn",
            "pi_benchmark": pi_method,
            "benchmark_method": "DAqrnn-star",
        },
        true_quantiles=true_q,
    ))

    deepquantreg_pred = None
    deepquantreg_boot = None
    deepquantreg_benchmark_pred = None
    deepquantreg_benchmark_boot = None
    deepquantreg_benchmark_lower = None
    deepquantreg_benchmark_upper = None
    cold_bench_pred = None
    cold_bench_boot = None
    if include_deepquantreg:
        from .baselines import (
            run_deepquantreg_adapter,
            run_deepquantreg_bootstrap_predictions,
        )

        try:
            dq_dir = raw_dir / f"deepquantreg_rep_{rep:04d}_s{scenario}_{error}_{censor_type}_{int(censor_rate*100)}"
            dq_dir.mkdir(parents=True, exist_ok=True)
            deepquantreg_pred = run_deepquantreg_adapter(
                train, test.X, target_taus, dq_dir / "point",
            )

            if int(deepquantreg_bootstrap_reps) > 0:
                deepquantreg_boot = run_deepquantreg_bootstrap_predictions(
                    train, test.X, target_taus,
                    dq_dir / "bootstrap",
                    B=int(deepquantreg_bootstrap_reps),
                    rng=rng,
                )
                dq_lower, dq_upper = interval_bounds(deepquantreg_boot)

                if deepquantreg_pi_benchmark == "own":
                    full_train = as_fully_observed(train)
                    deepquantreg_benchmark_pred = run_deepquantreg_adapter(
                        full_train, test.X, target_taus, dq_dir / "benchmark_point",
                    )
                    deepquantreg_benchmark_boot = run_deepquantreg_bootstrap_predictions(
                        full_train, test.X, target_taus,
                        dq_dir / "benchmark_bootstrap",
                        B=int(deepquantreg_bootstrap_reps),
                        rng=rng,
                    )
                    deepquantreg_benchmark_lower, deepquantreg_benchmark_upper = interval_bounds(
                        deepquantreg_benchmark_boot
                    )
                    dq_bench_pred = deepquantreg_benchmark_pred
                    dq_bench_lower = deepquantreg_benchmark_lower
                    dq_bench_upper = deepquantreg_benchmark_upper
                    dq_pi_label = "own"
                    dq_benchmark_method = "DeepQuantreg-star"
                elif deepquantreg_pi_benchmark == "shared":
                    dq_bench_pred = benchmark_pred
                    dq_bench_lower = bench_lower
                    dq_bench_upper = bench_upper
                    dq_pi_label = pi_method
                    dq_benchmark_method = "CS-QRNN-star"
                else:  # "matched"
                    cold_bench_pred, cold_bench_boot, cold_bench_lower, cold_bench_upper, cold_bench_rows = benchmark_predictions(
                        train, test.X, target_taus, J_benchmark, lam_benchmark, rng,
                        bootstrap_reps=int(deepquantreg_bootstrap_reps),
                        max_iter=max_iter,
                        backend=backend, device=device, torch_dtype=torch_dtype,
                        pi_method="independent",
                    )
                    bench_rows = list(bench_rows) + list(cold_bench_rows)
                    dq_bench_pred = cold_bench_pred
                    dq_bench_lower = cold_bench_lower
                    dq_bench_upper = cold_bench_upper
                    dq_pi_label = "independent"
                    dq_benchmark_method = "CS-QRNN-star-independent"

                metric_rows.extend(metric_rows_from_predictions(
                    test.y_true, list(target_taus),
                    deepquantreg_pred, dq_bench_pred,
                    dq_lower, dq_upper, dq_bench_lower, dq_bench_upper,
                    {
                        **base_meta,
                        "method": "DeepQuantreg",
                        "pi_benchmark": dq_pi_label,
                        "benchmark_method": dq_benchmark_method,
                    },
                    true_quantiles=true_q,
                ))
            else:
                # QL-only path: no PI bootstrap of DeepQuantreg.
                dq_benchmark_method = "CS-QRNN-star"
                dq_bench_pred_for_ql = benchmark_pred
                if deepquantreg_pi_benchmark == "own":
                    full_train = as_fully_observed(train)
                    deepquantreg_benchmark_pred = run_deepquantreg_adapter(
                        full_train, test.X, target_taus, dq_dir / "benchmark_point",
                    )
                    dq_bench_pred_for_ql = deepquantreg_benchmark_pred
                    dq_benchmark_method = "DeepQuantreg-star"
                for j, tau in enumerate(target_taus):
                    ql_dq = quantile_loss(test.y_true, deepquantreg_pred[:, j], tau)
                    ql_b = quantile_loss(test.y_true, dq_bench_pred_for_ql[:, j], tau)
                    metric_rows.append({
                        **base_meta,
                        "method": "DeepQuantreg",
                        "pi_benchmark": np.nan,
                        "benchmark_method": dq_benchmark_method,
                        "tau": tau_key(tau),
                        "ql_censored": ql_dq,
                        "ql_benchmark": ql_b,
                        "ql_ratio": ql_dq / ql_b if ql_b > 0 else np.nan,
                        "lower_censored_mean": np.nan,
                        "upper_censored_mean": np.nan,
                        "lower_benchmark_mean": float(bench_lower[:, j].mean()),
                        "upper_benchmark_mean": float(bench_upper[:, j].mean()),
                        "piw_censored": np.nan,
                        "piw_benchmark": float((bench_upper[:, j] - bench_lower[:, j]).mean()),
                        "piw_ratio": np.nan,
                        "coverage": np.nan,
                        "benchmark_coverage": np.nan,
                    })
        except Exception as exc:
            metric_rows.append({
                **base_meta,
                "method": "DeepQuantreg",
                "tau": np.nan,
                "ql_censored": np.nan,
                "error_message": str(exc),
            })

    save_npz(
        raw_dir / f"comparison_rep_{rep:04d}_s{scenario}_{error}_{censor_type}_{int(censor_rate*100)}.npz",
        X_test=test.X,
        y_test=test.y_true,
        X_train=train.X,
        y_train_true=train.y_true,
        y_train_obs=train.y_obs,
        delta=train.delta,
        L=train.L,
        R=train.R,
        target_taus=np.array(list(target_taus)),
        true_quantiles=true_q,
        tau_grid=tau_grid,
        benchmark_reps=np.array(bench_S),
        deepquantreg_bootstrap_reps=np.array(int(deepquantreg_bootstrap_reps)),
        # DA-CSQRNN
        pred_dacsqrnn=np.array([]) if da_cs is None else da_cs.final_predictions,
        pred_dacsqrnn_iter=np.array([]) if da_cs is None else da_cs.iteration_predictions,
        lower_dacsqrnn=np.array([]) if da_cs is None else da_cs.lower,
        upper_dacsqrnn=np.array([]) if da_cs is None else da_cs.upper,
        # DAqrnn (Huber)
        pred_daqrnn=da_huber.final_predictions,
        pred_daqrnn_iter=da_huber.iteration_predictions,
        lower_daqrnn=da_huber.lower,
        upper_daqrnn=da_huber.upper,
        # Warm-start benchmark
        pred_benchmark=benchmark_pred,
        pred_benchmark_boot=benchmark_boot,
        lower_benchmark=bench_lower,
        upper_benchmark=bench_upper,
        # DAqrnn fully observed benchmark
        pred_benchmark_daqrnn=benchmark_huber_pred,
        pred_benchmark_daqrnn_boot=benchmark_huber_boot,
        lower_benchmark_daqrnn=huber_bench_lower,
        upper_benchmark_daqrnn=huber_bench_upper,
        # DeepQuantreg
        pred_deepquantreg=np.array([]) if deepquantreg_pred is None else deepquantreg_pred,
        pred_deepquantreg_boot=np.array([]) if deepquantreg_boot is None else deepquantreg_boot,
        pred_deepquantreg_benchmark=np.array([]) if deepquantreg_benchmark_pred is None else deepquantreg_benchmark_pred,
        pred_deepquantreg_benchmark_boot=np.array([]) if deepquantreg_benchmark_boot is None else deepquantreg_benchmark_boot,
        lower_deepquantreg_benchmark=np.array([]) if deepquantreg_benchmark_lower is None else deepquantreg_benchmark_lower,
        upper_deepquantreg_benchmark=np.array([]) if deepquantreg_benchmark_upper is None else deepquantreg_benchmark_upper,
        # Cold-start benchmark (only populated when matched)
        pred_benchmark_cold=np.array([]) if cold_bench_pred is None else cold_bench_pred,
        pred_benchmark_cold_boot=np.array([]) if cold_bench_boot is None else cold_bench_boot,
        lower_benchmark_cold=np.array([]) if cold_bench_pred is None else cold_bench_lower,
        upper_benchmark_cold=np.array([]) if cold_bench_pred is None else cold_bench_upper,
    )

    fit_rows = ([] if da_cs is None else list(da_cs.fit_rows)) + list(da_huber.fit_rows) + list(bench_rows)
    for row in fit_rows:
        if not isinstance(row, dict):
            continue
        row.update({k: base_meta[k] for k in ("rep", "scenario", "error", "censor_type", "censor_rate")})
    return metric_rows, fit_rows
