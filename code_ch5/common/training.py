"""Model fitting, EBIC search, and bootstrap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import J_GRID, LAMBDA_GRID
from .model import QRNNModel, loss_and_grad, loss_values, pack_params, parameter_count, unpack_params


@dataclass
class FitInfo:
    converged: bool
    objective: float
    nit: int
    message: str


def tau_key(tau: float) -> float:
    return float(np.round(float(tau), 6))


def resolve_hyperparameter(value, tau: float):
    """Return a scalar hyperparameter, using nearest tau when maps lack random DA taus."""

    if not isinstance(value, dict):
        return value
    key = tau_key(tau)
    if key in value:
        return value[key]
    if not value:
        raise KeyError(f"empty hyperparameter map for tau={tau}")
    keys = np.array(list(value.keys()), dtype=float)
    nearest = float(keys[np.argmin(np.abs(keys - key))])
    return value[nearest]


def x_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale <= 1e-8, 1.0, scale)
    return mean, scale


def initial_params(p: int, J: int, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = rng.normal(0.0, 0.15, size=(p, J))
    b = np.zeros(J)
    v = rng.normal(0.0, 0.15, size=J)
    c = float(np.median(y))
    return pack_params(W, b, v, c)


def effective_df(model: QRNNModel, mode: str = "selected_variables", active_threshold: float = 1e-4) -> int:
    """Effective degrees of freedom for the EBIC-like criterion.

    Hao et al. define ``df`` as the number of selected variables corresponding
    to lambda and J, not as the total number of QRNN weights. The default mode
    follows that definition. Other modes are provided for sensitivity checks.
    """

    W, _, v, _ = unpack_params(model.params, model.p, model.J)
    threshold = float(active_threshold)
    if mode == "selected_variables":
        active = np.linalg.norm(W, axis=1) > threshold
        return int(max(1, active.sum()))
    if mode == "active_hidden_nodes":
        active = (np.linalg.norm(W, axis=0) > threshold) & (np.abs(v) > threshold)
        return int(max(1, active.sum()))
    if mode == "hidden_weights":
        return int(max(1, np.sum(np.abs(W) > threshold)))
    if mode == "parameters":
        return parameter_count(model.p, model.J)
    raise ValueError(f"unknown df_mode: {mode}")


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    J: int,
    lam: float,
    rng: np.random.Generator,
    h: float | None = None,
    max_iter: int = 250,
    init_model: QRNNModel | None = None,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    correction: np.ndarray | None = None,
    fixed_x_stats: tuple[np.ndarray, np.ndarray] | None = None,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[QRNNModel, FitInfo]:
    """Fit one QRNN by full-batch Adam, optionally on a PyTorch device."""

    if len(y) == 0:
        raise ValueError("cannot fit QRNN with zero observations")
    p = int(X.shape[1])
    h = float(0.1 * np.std(y) if h is None else h)
    h = max(h, 1e-4)

    if init_model is not None and init_model.J == J and init_model.p == p:
        mean, scale = init_model.x_mean, init_model.x_scale
        start = init_model.params.copy()
        h = init_model.h if h is None else h
    else:
        mean, scale = fixed_x_stats if fixed_x_stats is not None else x_stats(X)
        start = initial_params(p, J, y, rng)

    Xs = (X - mean) / scale

    if backend in {"auto", "torch"} or device not in {"cpu", "numpy"}:
        from .torch_backend import fit_torch, resolve_device

        resolved_backend, resolved_device = resolve_device(device=device, backend=backend)
        if resolved_backend == "torch":
            result = fit_torch(
                Xs,
                y,
                start,
                p,
                J,
                tau,
                lam,
                h,
                loss_kind,
                epsilon,
                correction,
                max_iter,
                resolved_device,
                dtype_name=torch_dtype,
            )
            model = QRNNModel(
                p=p,
                J=int(J),
                tau=float(tau),
                lam=float(lam),
                h=float(h),
                params=result.params,
                x_mean=np.asarray(mean, dtype=float),
                x_scale=np.asarray(scale, dtype=float),
                loss_kind=loss_kind,
                epsilon=float(epsilon),
                backend="torch",
                device=resolved_device,
                torch_dtype=torch_dtype,
            )
            info = FitInfo(result.converged, result.objective, result.nit, result.message)
            return model, info

    def fun(params: np.ndarray) -> tuple[float, np.ndarray]:
        obj, grad = loss_and_grad(
            params,
            Xs,
            y,
            p,
            J,
            tau,
            lam,
            h,
            loss_kind=loss_kind,
            epsilon=epsilon,
            correction=correction,
            include_penalty=True,
        )
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            return 1e30, np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        return obj, grad

    params = start.astype(float).copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best_params = params.copy()
    best_obj = np.inf
    last_obj: float | None = None
    converged = False
    message = "max_iter reached"
    lr = 0.03
    beta1, beta2 = 0.9, 0.999
    eps_adam = 1e-8
    nit = 0
    for t in range(1, int(max_iter) + 1):
        nit = t
        obj, grad = fun(params)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 100.0:
            grad = grad * (100.0 / grad_norm)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
        if last_obj is not None and abs(last_obj - obj) <= 1e-7 * (1.0 + abs(last_obj)):
            converged = True
            message = "relative objective tolerance reached"
            break
        last_obj = obj
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        step = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        params = params - step
        if np.linalg.norm(step) <= 1e-7 * (1.0 + np.linalg.norm(params)):
            converged = True
            message = "step tolerance reached"
            break

    params = best_params
    model = QRNNModel(
        p=p,
        J=int(J),
        tau=float(tau),
        lam=float(lam),
        h=float(h),
        params=params,
        x_mean=np.asarray(mean, dtype=float),
        x_scale=np.asarray(scale, dtype=float),
        loss_kind=loss_kind,
        epsilon=float(epsilon),
        backend="numpy",
        device="cpu",
        torch_dtype=torch_dtype,
    )
    info = FitInfo(bool(converged), float(best_obj), int(nit), message)
    return model, info


def fit_models_for_taus(
    X: np.ndarray,
    y: np.ndarray,
    taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    rng: np.random.Generator,
    max_iter: int,
    warm_starts: dict[float, QRNNModel] | None = None,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[dict[float, QRNNModel], list[dict[str, object]]]:
    models: dict[float, QRNNModel] = {}
    rows: list[dict[str, object]] = []
    warm_starts = warm_starts or {}
    sorted_taus = sorted({tau_key(t) for t in taus})

    can_use_torch = (backend in {"auto", "torch"}) or (device not in {"cpu", "numpy"})
    if can_use_torch:
        from .torch_backend import fit_torch_batch, resolve_device

        resolved_backend, resolved_device = resolve_device(device=device, backend=backend)
        if resolved_backend == "torch":
            p = int(X.shape[1])
            h_default = max(float(0.1 * np.std(y)), 1e-4)
            current_mean, current_scale = x_stats(X)

            taus_by_J: dict[int, list[float]] = {}
            for tau in sorted_taus:
                J_tau = int(resolve_hyperparameter(J, tau))
                taus_by_J.setdefault(J_tau, []).append(tau)

            for J_val, tau_group in sorted(taus_by_J.items()):
                # Warm starts live in the standardized coordinate system of
                # the model that produced them. Reusing their raw parameters
                # with freshly computed bootstrap-sample x_stats changes that
                # coordinate system and is not a true warm start. When a batch
                # has compatible warm starts, keep their x_stats; otherwise
                # fall back to the current sample's x_stats.
                warm_inits = [
                    warm_starts.get(tau)
                    for tau in tau_group
                    if warm_starts.get(tau) is not None
                    and warm_starts[tau].J == J_val
                    and warm_starts[tau].p == p
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
                Xs = (X - mean) / scale

                starts = []
                tau_list = []
                lam_list = []
                h_list = []
                for tau in tau_group:
                    lam_tau = float(resolve_hyperparameter(lam, tau))
                    init = warm_starts.get(tau)
                    if init is not None and init.J == J_val and init.p == p:
                        starts.append(init.params.copy())
                        h_list.append(init.h)
                    else:
                        starts.append(initial_params(p, J_val, y, rng))
                        h_list.append(h_default)
                    tau_list.append(float(tau))
                    lam_list.append(lam_tau)

                results = fit_torch_batch(
                    Xs, y, starts, p, J_val, tau_list, lam_list, h_list,
                    loss_kind, epsilon, max_iter, resolved_device,
                    dtype_name=torch_dtype,
                )
                for tau, result, h_val, lam_tau in zip(tau_group, results, h_list, lam_list):
                    model = QRNNModel(
                        p=p, J=J_val, tau=float(tau), lam=float(lam_tau),
                        h=float(h_val), params=result.params,
                        x_mean=np.asarray(mean, dtype=float),
                        x_scale=np.asarray(scale, dtype=float),
                        loss_kind=loss_kind, epsilon=float(epsilon),
                        backend="torch", device=resolved_device,
                        torch_dtype=torch_dtype,
                    )
                    models[tau] = model
                    rows.append({
                        "tau": tau, "J": J_val, "lambda": lam_tau,
                        "converged": result.converged, "objective": result.objective,
                        "nit": result.nit, "message": result.message,
                        "backend": "torch", "device": resolved_device,
                    })
            return models, rows

    for tau in sorted_taus:
        J_tau = int(resolve_hyperparameter(J, tau))
        lam_tau = float(resolve_hyperparameter(lam, tau))
        init = warm_starts.get(tau)
        model, info = fit_model(
            X,
            y,
            tau,
            J_tau,
            lam_tau,
            rng,
            max_iter=max_iter,
            init_model=init,
            loss_kind=loss_kind,
            epsilon=epsilon,
            backend=backend,
            device=device,
            torch_dtype=torch_dtype,
        )
        models[tau] = model
        rows.append(
            {
                "tau": tau,
                "J": J_tau,
                "lambda": lam_tau,
                "converged": info.converged,
                "objective": info.objective,
                "nit": info.nit,
                "message": info.message,
                "backend": model.backend,
                "device": model.device,
            }
        )
    return models, rows


def select_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    J_grid: Iterable[int] = J_GRID,
    lambda_grid: Iterable[float] = LAMBDA_GRID,
    gamma: float = 0.5,
    max_iter: int = 120,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    df_mode: str = "selected_variables",
    active_threshold: float = 1e-4,
    n_restarts: int = 3,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search EBIC for one tau."""

    rows: list[dict[str, object]] = []
    p = int(X.shape[1])
    n = int(len(y))
    J_list = list(J_grid)
    lam_list = list(lambda_grid)
    combos = [(j, l) for j in J_list for l in lam_list]
    for J, lam in tqdm(combos, desc=f"EBIC tau={tau:.2f}", leave=False):
        candidates = []
        for restart in range(int(max(1, n_restarts))):
            model, info = fit_model(
                X,
                y,
                tau,
                int(J),
                float(lam),
                rng,
                max_iter=max_iter,
                loss_kind=loss_kind,
                epsilon=epsilon,
                backend=backend,
                device=device,
                torch_dtype=torch_dtype,
            )
            pred = model.predict(X)
            mean_loss = float(loss_values(y - pred, tau, model.h, loss_kind, epsilon).mean())
            candidates.append((mean_loss, restart, model, info))
        mean_loss, restart, model, info = min(candidates, key=lambda item: item[0])
        pred = model.predict(X)
        mean_loss = float(loss_values(y - pred, tau, model.h, loss_kind, epsilon).mean())
        df = effective_df(model, mode=df_mode, active_threshold=active_threshold)
        ebic = np.log(max(mean_loss, 1e-12)) + (np.log(n) / n) * df + gamma * (np.log(p) / n) * df
        rows.append(
            {
                "tau": tau,
                "J": int(J),
                "lambda": float(lam),
                "h": model.h,
                "df": df,
                "df_mode": df_mode,
                "active_threshold": float(active_threshold),
                "best_restart": int(restart),
                "n_restarts": int(max(1, n_restarts)),
                "mean_loss": mean_loss,
                "ebic": float(ebic),
                "converged": info.converged,
                "nit": info.nit,
                "backend": model.backend,
                "device": model.device,
            }
        )
    grid = pd.DataFrame(rows).sort_values("ebic").reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    return {
        "J": int(best["J"]),
        "lambda": float(best["lambda"]),
        "ebic": float(best["ebic"]),
        "mean_loss": float(best["mean_loss"]),
        "df": int(best["df"]),
        "df_mode": str(best["df_mode"]),
        "best_restart": int(best["best_restart"]),
        "n_restarts": int(best["n_restarts"]),
    }, grid


def bootstrap_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    taus: Iterable[float],
    J: int | dict[float, int],
    lam: float | dict[float, float],
    rng: np.random.Generator,
    B: int,
    max_iter: int,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return raw bootstrap predictions with shape (B, n_test, n_taus)."""

    taus = [tau_key(t) for t in taus]
    preds = np.empty((int(B), X_test.shape[0], len(taus)), dtype=float)
    rows: list[dict[str, object]] = []
    n = len(y_train)
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        models, fit_rows = fit_models_for_taus(
            X_train[idx],
            y_train[idx],
            taus,
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
        for j, tau in enumerate(taus):
            preds[b, :, j] = models[tau].predict(X_test)
        for row in fit_rows:
            row = dict(row)
            row["bootstrap"] = b
            rows.append(row)
    return preds, rows
