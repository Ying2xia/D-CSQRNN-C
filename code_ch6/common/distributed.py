"""Distributed DCS-QRNN-C simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from .augmentation import DAResult
from .data import SimDataset, boundary_values, impute_from_candidate_predictions, make_tau_grid, subset_dataset
from .metrics import interval_bounds
from .model import QRNNModel
from .training import fit_model, initial_params, resolve_hyperparameter, tau_key, x_stats


@dataclass
class DistributedResult(DAResult):
    elapsed_seconds: float = 0.0
    communication_rounds: int = 0


def partition_dataset(dataset: SimDataset, K: int, rng: np.random.Generator) -> list[SimDataset]:
    idx = rng.permutation(dataset.n)
    return [subset_dataset(dataset, part) for part in np.array_split(idx, K)]


def _average_qrnn_models(models: list[QRNNModel]) -> QRNNModel:
    """Average worker QRNN model fields directly for the OS baseline."""

    if not models:
        raise ValueError("cannot average zero QRNN models")
    ref = models[0]
    for model in models[1:]:
        if model.p != ref.p or model.J != ref.J:
            raise ValueError("all worker models must share p and J before parameter averaging")
        if tau_key(model.tau) != tau_key(ref.tau):
            raise ValueError("all worker models must share tau before parameter averaging")
    params = np.mean([model.params for model in models], axis=0)
    x_mean = np.mean([model.x_mean for model in models], axis=0)
    x_scale = np.mean([model.x_scale for model in models], axis=0)
    x_scale = np.where(x_scale <= 1e-8, 1.0, x_scale)
    h = float(np.mean([model.h for model in models]))
    lam = float(np.mean([model.lam for model in models]))
    return QRNNModel(
        p=ref.p,
        J=ref.J,
        tau=float(ref.tau),
        lam=lam,
        h=h,
        params=np.asarray(params, dtype=float),
        x_mean=np.asarray(x_mean, dtype=float),
        x_scale=np.asarray(x_scale, dtype=float),
        loss_kind=ref.loss_kind,
        epsilon=ref.epsilon,
        backend=ref.backend,
        device=ref.device,
        torch_dtype=ref.torch_dtype,
    )


def _fit_surrogate_round(
    partitions_X: list[np.ndarray],
    partitions_y: list[np.ndarray],
    tau: float,
    J: int,
    lam: float,
    rng: np.random.Generator,
    pilot,
    max_iter: int,
    loss_kind: str,
    epsilon: float,
    backend: str,
    device: str,
    torch_dtype: str,
):
    grads = []
    weights = []
    for Xk, yk in zip(partitions_X, partitions_y):
        grads.append(pilot.empirical_gradient(Xk, yk))
        weights.append(len(yk))
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    global_grad = np.sum([w * g for w, g in zip(weights_arr, grads)], axis=0)
    local_grad = grads[0]
    correction = local_grad - global_grad
    model, info = fit_model(
        partitions_X[0],
        partitions_y[0],
        tau,
        J,
        lam,
        rng,
        h=pilot.h,
        max_iter=max_iter,
        init_model=pilot,
        loss_kind=loss_kind,
        epsilon=epsilon,
        correction=correction,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    return model, info


def fit_dcs_tau(
    partitions_X: list[np.ndarray],
    partitions_y: list[np.ndarray],
    tau: float,
    J: int,
    lam: float,
    rng: np.random.Generator,
    R: int,
    max_iter: int,
    warm_start=None,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[object, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    if warm_start is None:
        pilot, info = fit_model(
            partitions_X[0],
            partitions_y[0],
            tau,
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
        rows.append(
            {
                "tau": tau,
                "ilea_round": 0,
                "stage": "pilot",
                "converged": info.converged,
                "objective": info.objective,
                "nit": info.nit,
                "message": info.message,
            }
        )
    else:
        pilot = warm_start

    model = pilot
    for r in range(int(R)):
        model, info = _fit_surrogate_round(
            partitions_X,
            partitions_y,
            tau,
            J,
            lam,
            rng,
            pilot=model,
            max_iter=max_iter,
            loss_kind=loss_kind,
            epsilon=epsilon,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        rows.append(
            {
                "tau": tau,
                "ilea_round": r + 1,
                "stage": "surrogate_update",
                "converged": info.converged,
                "objective": info.objective,
                "nit": info.nit,
                "message": info.message,
            }
        )
    return model, rows


def _torch_batch_tools(backend: str, device: str):
    """Return PyTorch batch fitting tools when the requested runtime supports them."""

    if not ((backend in {"auto", "torch"}) or (device not in {"cpu", "numpy"})):
        return None
    try:
        from .torch_backend import fit_torch_batch, resolve_device

        resolved_backend, resolved_device = resolve_device(device=device, backend=backend)
    except RuntimeError:
        if backend == "auto":
            return None
        raise
    if resolved_backend != "torch":
        return None
    return fit_torch_batch, resolved_device


def fit_dcs_taus(
    partitions_X: list[np.ndarray],
    partitions_y: list[np.ndarray],
    taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    rng: np.random.Generator,
    R: int,
    max_iter: int,
    warm_starts: dict[float, object] | None = None,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[dict[float, object], list[dict[str, object]]]:
    """Fit distributed models for several tau values, batching same-J fits on GPU."""

    tau_list = sorted({tau_key(t) for t in taus})
    warm_starts = warm_starts or {}
    batch_tools = _torch_batch_tools(backend, device)
    if batch_tools is None:
        models: dict[float, object] = {}
        rows: list[dict[str, object]] = []
        for tau in tau_list:
            model, tau_rows = fit_dcs_tau(
                partitions_X,
                partitions_y,
                tau,
                int(resolve_hyperparameter(J, tau)),
                float(resolve_hyperparameter(lam, tau)),
                rng,
                R=R,
                max_iter=max_iter,
                warm_start=warm_starts.get(tau),
                loss_kind=loss_kind,
                epsilon=epsilon,
                backend=backend,
                device=device,
                torch_dtype=torch_dtype,
            )
            models[tau] = model
            rows.extend(tau_rows)
        return models, rows

    fit_torch_batch, resolved_device = batch_tools
    X0 = partitions_X[0]
    y0 = partitions_y[0]
    p = int(X0.shape[1])
    current_mean, current_scale = x_stats(X0)
    h_default = max(float(0.1 * np.std(y0)), 1e-4)
    models: dict[float, object] = {}
    rows: list[dict[str, object]] = []

    taus_by_J: dict[int, list[float]] = {}
    for tau in tau_list:
        J_tau = int(resolve_hyperparameter(J, tau))
        taus_by_J.setdefault(J_tau, []).append(tau)

    for J_val, group in sorted(taus_by_J.items()):
        warm_inits = [
            warm_starts.get(tau)
            for tau in group
            if warm_starts.get(tau) is not None
            and getattr(warm_starts[tau], "J", None) == J_val
            and getattr(warm_starts[tau], "p", None) == p
        ]
        if warm_inits:
            ref_mean = warm_inits[0].x_mean
            ref_scale = warm_inits[0].x_scale
            same_stats = all(
                np.allclose(init.x_mean, ref_mean) and np.allclose(init.x_scale, ref_scale)
                for init in warm_inits
            )
            mean, scale = (ref_mean, ref_scale) if same_stats else (current_mean, current_scale)
        else:
            mean, scale = current_mean, current_scale
        X0s = (X0 - mean) / scale

        missing = [
            tau
            for tau in group
            if warm_starts.get(tau) is None
            or getattr(warm_starts[tau], "J", None) != J_val
            or getattr(warm_starts[tau], "p", None) != p
        ]
        if missing:
            starts = [initial_params(p, J_val, y0, rng) for _ in missing]
            lams = [float(resolve_hyperparameter(lam, tau)) for tau in missing]
            hs = [h_default for _ in missing]
            results = fit_torch_batch(
                X0s,
                y0,
                starts,
                p,
                J_val,
                [float(tau) for tau in missing],
                lams,
                hs,
                loss_kind,
                epsilon,
                max_iter,
                resolved_device,
                dtype_name=torch_dtype,
            )
            for tau, result, h_val, lam_tau in zip(missing, results, hs, lams):
                models[tau] = QRNNModel(
                    p=p,
                    J=J_val,
                    tau=float(tau),
                    lam=float(lam_tau),
                    h=float(h_val),
                    params=result.params,
                    x_mean=np.asarray(mean, dtype=float),
                    x_scale=np.asarray(scale, dtype=float),
                    loss_kind=loss_kind,
                    epsilon=float(epsilon),
                    backend="torch",
                    device=resolved_device,
                    torch_dtype=torch_dtype,
                )
                rows.append(
                    {
                        "tau": tau,
                        "ilea_round": 0,
                        "stage": "pilot",
                        "converged": result.converged,
                        "objective": result.objective,
                        "nit": result.nit,
                        "message": result.message,
                        "backend": "torch",
                        "device": resolved_device,
                    }
                )

        for tau in group:
            if tau not in models:
                models[tau] = warm_starts[tau]

        for r in range(int(R)):
            starts = []
            lams = []
            hs = []
            corrections = []
            for tau in group:
                model = models[tau]
                grads = []
                weights = []
                for Xk, yk in zip(partitions_X, partitions_y):
                    grads.append(model.empirical_gradient(Xk, yk))
                    weights.append(len(yk))
                weights_arr = np.asarray(weights, dtype=float)
                weights_arr /= weights_arr.sum()
                global_grad = np.sum([w * g for w, g in zip(weights_arr, grads)], axis=0)
                correction = grads[0] - global_grad
                starts.append(model.params.copy())
                lams.append(float(resolve_hyperparameter(lam, tau)))
                hs.append(float(model.h))
                corrections.append(correction)

            results = fit_torch_batch(
                X0s,
                y0,
                starts,
                p,
                J_val,
                [float(tau) for tau in group],
                lams,
                hs,
                loss_kind,
                epsilon,
                max_iter,
                resolved_device,
                dtype_name=torch_dtype,
                corrections=corrections,
            )
            for tau, result, h_val, lam_tau in zip(group, results, hs, lams):
                models[tau] = QRNNModel(
                    p=p,
                    J=J_val,
                    tau=float(tau),
                    lam=float(lam_tau),
                    h=float(h_val),
                    params=result.params,
                    x_mean=np.asarray(mean, dtype=float),
                    x_scale=np.asarray(scale, dtype=float),
                    loss_kind=loss_kind,
                    epsilon=float(epsilon),
                    backend="torch",
                    device=resolved_device,
                    torch_dtype=torch_dtype,
                )
                rows.append(
                    {
                        "tau": tau,
                        "ilea_round": r + 1,
                        "stage": "surrogate_update",
                        "converged": result.converged,
                        "objective": result.objective,
                        "nit": result.nit,
                        "message": result.message,
                        "backend": "torch",
                        "device": resolved_device,
                    }
                )
    return models, rows


def run_dcs_qrnn_c(
    partitions: list[SimDataset],
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    S: int,
    R: int,
    tau_grid: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 120,
    burn_in: int = 0,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> DistributedResult:
    import time

    start = time.perf_counter()
    target_taus = [tau_key(t) for t in target_taus]
    grid_taus = [tau_key(t) for t in tau_grid]
    full_taus = sorted(set(grid_taus).union(target_taus))
    S = int(S)
    burn_in = int(min(max(burn_in, 0), max(S - 1, 0)))
    current_y = [boundary_values(part) for part in partitions]
    models: dict[float, object] = {}
    fit_rows: list[dict[str, object]] = []
    iter_preds = np.empty((S, X_test.shape[0], len(target_taus)), dtype=float)

    X_src = [part.X[part.uncensored_mask] for part in partitions]
    y_src = [part.y_obs[part.uncensored_mask] for part in partitions]
    if len(y_src[0]) < 10:
        X_src = [part.X for part in partitions]
        y_src = current_y
    init_models, rows = fit_dcs_taus(
        X_src,
        y_src,
        full_taus,
        J,
        lam,
        rng,
        R=max(1, int(R)),
        max_iter=max_iter,
        loss_kind=loss_kind,
        epsilon=epsilon,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )
    models.update(init_models)
    for row in rows:
        row.update({"da_iteration": 0, "stage_outer": "initial"})
        fit_rows.append(row)

    for s in range(S):
        current_y = [
            impute_from_candidate_predictions(
                np.column_stack([models[tau].predict(part.X) for tau in full_taus]),
                part,
                rng,
            )
            for part in partitions
        ]
        boot_X: list[np.ndarray] = []
        boot_y: list[np.ndarray] = []
        for part, yk in zip(partitions, current_y):
            idx = rng.integers(0, part.n, size=part.n)
            boot_X.append(part.X[idx])
            boot_y.append(yk[idx])

        updated, rows = fit_dcs_taus(
            boot_X,
            boot_y,
            full_taus,
            J,
            lam,
            rng,
            R=R,
            max_iter=max_iter,
            warm_starts={tau: models[tau] for tau in full_taus if tau in models},
            loss_kind=loss_kind,
            epsilon=epsilon,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        models.update(updated)
        for row in rows:
            row.update({"da_iteration": s + 1, "stage_outer": "target_update"})
            fit_rows.append(row)

        for j, tau in enumerate(target_taus):
            iter_preds[s, :, j] = models[tau].predict(X_test)

    kept = iter_preds[burn_in:] if burn_in < S else iter_preds
    final = kept.mean(axis=0)
    lower, upper = interval_bounds(kept)
    elapsed = time.perf_counter() - start
    return DistributedResult(
        target_taus=target_taus,
        tau_schedule=np.array([], dtype=float),
        final_predictions=final,
        iteration_predictions=iter_preds,
        lower=lower,
        upper=upper,
        last_imputed_y=np.concatenate(current_y),
        fit_rows=fit_rows,
        final_models={tau: models[tau] for tau in target_taus},
        elapsed_seconds=float(elapsed),
        communication_rounds=int(S * R * len(full_taus)),
    )


def run_one_shot(
    partitions: list[SimDataset],
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    S: int,
    tau_grid: np.ndarray,
    rng: np.random.Generator,
    max_iter: int,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    """Run the OS baseline with host-side network-parameter averaging.

    Each worker only observes its local partition, so its DA quantile grid is
    generated from ``part.n``. The host then directly averages the final worker
    QRNN parameter vectors and predicts from the averaged model.
    """

    from .augmentation import run_da_csqrnn

    target_taus = [tau_key(t) for t in target_taus]
    worker_models: dict[float, list[QRNNModel]] = {tau: [] for tau in target_taus}
    rows = []
    for k, part in enumerate(partitions):
        worker_tau_grid = make_tau_grid(part.n)
        res = run_da_csqrnn(
            part,
            X_test,
            target_taus,
            J,
            lam,
            S,
            worker_tau_grid,
            rng,
            max_iter=max_iter,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        for tau in target_taus:
            worker_models[tau].append(res.final_models[tau])
        for row in res.fit_rows:
            row = dict(row)
            row["worker"] = k + 1
            row["worker_n"] = part.n
            row["tau_grid_n"] = len(worker_tau_grid)
            rows.append(row)

    averaged_pred = np.empty((X_test.shape[0], len(target_taus)), dtype=float)
    for j, tau in enumerate(target_taus):
        averaged_model = _average_qrnn_models(worker_models[tau])
        averaged_pred[:, j] = averaged_model.predict(X_test)
    return averaged_pred, rows
