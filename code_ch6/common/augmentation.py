"""Centralized DA-CSQRNN and DAqrnn-style data augmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .data import SimDataset, boundary_values, impute_from_candidate_predictions
from .metrics import interval_bounds
from .training import fit_models_for_taus, tau_key


@dataclass
class DAResult:
    target_taus: list[float]
    tau_schedule: np.ndarray
    final_predictions: np.ndarray
    iteration_predictions: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    last_imputed_y: np.ndarray
    fit_rows: list[dict[str, object]]
    final_models: dict[float, object] = field(default_factory=dict)


def _training_source(dataset: SimDataset, current_y: np.ndarray, prefer_uncensored: bool) -> tuple[np.ndarray, np.ndarray]:
    mask = dataset.uncensored_mask
    if prefer_uncensored and int(mask.sum()) >= 10:
        return dataset.X[mask], dataset.y_obs[mask]
    return dataset.X, current_y


def run_da_csqrnn(
    dataset: SimDataset,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    S: int,
    tau_grid: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 250,
    burn_in: int = 0,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> DAResult:
    """Run the DA algorithm for censored QRNN estimation.

    This follows the paper algorithm: every DA iteration imputes from the full
    quantile grid and then updates all grid models.

    Parameters
    ----------
    dataset
        Censored training data, including observed responses, true responses
        for simulation evaluation, censoring indicators, and censoring bounds.
    X_test
        Test predictors where conditional quantiles are predicted.
    target_taus
        Quantile levels reported in the Chapter 5 tables, usually
        ``(0.10, 0.25, 0.50, 0.75, 0.90)``.
    J
        Number of hidden-layer nodes. May be one scalar or a per-tau map from
        the Section 5.4.1 EBIC search.
    lam
        Regularization strength. May be one scalar or a per-tau map.
    S
        Number of DA iterations used to form the averaged prediction and PI.
    tau_grid
        Quantile grid used to impute censored responses. In full-grid mode all
        grid models are available for the imputation draw in each DA iteration.
    rng
        NumPy random generator controlling tau draws, imputations, bootstrap
        samples, and model initialization.
    max_iter
        Adam optimization steps for each QRNN fit.
    burn_in
        Number of initial DA iterations to discard before averaging and
        building the prediction interval.
    loss_kind
        QRNN loss type: ``"cs"`` for DA-CSQRNN, ``"huber"`` for DAqrnn, or
        ``"check"`` for unsmoothed quantile loss.
    epsilon
        Huber smoothing parameter, used only when ``loss_kind="huber"``.
    backend, device, torch_dtype
        Compute backend controls. Use ``backend="numpy"`` for CPU NumPy, or
        ``backend="auto"`` / ``"torch"`` with a CUDA/MPS device when available.
    """

    target_taus = [tau_key(t) for t in target_taus]
    grid_taus = [tau_key(t) for t in tau_grid]
    full_taus = sorted(set(grid_taus).union(target_taus))
    S = int(S)
    burn_in = int(min(max(burn_in, 0), max(S - 1, 0)))
    current_y = boundary_values(dataset)
    models: dict[float, object] = {}
    fit_rows: list[dict[str, object]] = []

    X0, y0 = _training_source(dataset, current_y, prefer_uncensored=True)
    init_models, rows = fit_models_for_taus(
        X0,
        y0,
        full_taus,
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
    models.update(init_models)
    for row in rows:
        row.update({"da_iteration": 0, "stage": "initial"})
        fit_rows.append(row)

    iter_preds = np.empty((S, X_test.shape[0], len(target_taus)), dtype=float)
    for s in range(S):
        candidate_pred = np.column_stack([models[tau].predict(dataset.X) for tau in full_taus])
        current_y = impute_from_candidate_predictions(candidate_pred, dataset, rng)
        boot_idx = rng.integers(0, dataset.n, size=dataset.n)
        X_boot, y_boot = dataset.X[boot_idx], current_y[boot_idx]

        updated, rows = fit_models_for_taus(
            X_boot,
            y_boot,
            full_taus,
            J,
            lam,
            rng,
            max_iter=max_iter,
            warm_starts={t: models[t] for t in full_taus if t in models},
            loss_kind=loss_kind,
            epsilon=epsilon,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        models.update(updated)
        for row in rows:
            row.update({"da_iteration": s + 1, "stage": "update"})
            fit_rows.append(row)

        for j, tau in enumerate(target_taus):
            iter_preds[s, :, j] = models[tau].predict(X_test)

    kept = iter_preds[burn_in:] if burn_in < S else iter_preds
    final = kept.mean(axis=0)
    lower, upper = interval_bounds(kept)
    return DAResult(
        target_taus=target_taus,
        tau_schedule=np.array([], dtype=float),
        final_predictions=final,
        iteration_predictions=iter_preds,
        lower=lower,
        upper=upper,
        last_imputed_y=current_y,
        fit_rows=fit_rows,
        final_models={tau: models[tau] for tau in target_taus},
    )
