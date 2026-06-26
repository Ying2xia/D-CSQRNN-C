"""Baseline method adapters used in Section 5.4.3."""

from __future__ import annotations

import os
import shlex
import sys
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from .augmentation import run_da_csqrnn
from .data import SimDataset


def _deepquantreg_command() -> list[str]:
    """Return an external DeepQuantreg command, defaulting to the bundled runner."""

    cmd = os.environ.get("DEEPQUANTREG_CMD")
    if cmd:
        return shlex.split(cmd)
    runner = Path(__file__).resolve().parents[1] / "scripts" / "deepquantreg_runner.py"
    if runner.exists():
        return [sys.executable, str(runner)]
    raise RuntimeError(
        "DeepQuantreg runner was not found. Put scripts/deepquantreg_runner.py in the project "
        "or set DEEPQUANTREG_CMD."
    )


def _deepquantreg_runner_options(include_defaults: bool = False) -> list[str]:
    """Optional CLI overrides for the bundled DeepQuantreg runner.

    Environment variables keep Section 5.4.3 reproducible while allowing quick
    runtime sensitivity checks without editing the Python source.
    """

    options: list[str] = []
    env_to_flag = {
        "DEEPQUANTREG_EPOCHS": ("--epochs", "200"),
        "DEEPQUANTREG_LAYER": ("--layer", "2"),
        "DEEPQUANTREG_NODE": ("--node", "10"),
        "DEEPQUANTREG_BATCH_SIZE": ("--batch-size", "64"),
    }
    for env_name, (flag, default) in env_to_flag.items():
        value = os.environ.get(env_name)
        if include_defaults and not value:
            value = default
        if value:
            options.extend([flag, value])
    return options


def run_daqrnn_baseline(
    dataset: SimDataset,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    J,
    lam,
    S: int,
    tau_grid: np.ndarray,
    rng: np.random.Generator,
    max_iter: int,
    epsilon: float = 0.1,
    backend: str = "numpy",
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    """DAqrnn-style baseline: same DA loop, Huber-smoothed QRNN loss."""

    return run_da_csqrnn(
        dataset,
        X_test,
        target_taus,
        J,
        lam,
        S,
        tau_grid,
        rng,
        max_iter=max_iter,
        loss_kind="huber",
        epsilon=epsilon,
        backend=backend,
        device=device,
        torch_dtype=torch_dtype,
    )


def run_deepquantreg_adapter(
    dataset: SimDataset,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    output_dir: Path,
) -> np.ndarray:
    """Adapter for an external DeepQuantreg implementation.

    By default this calls ``scripts/deepquantreg_runner.py`` with the current
    Python interpreter. Set DEEPQUANTREG_CMD only if you want to override that
    command. The prediction file must be an npz containing ``pred`` with shape
    ``(n_test, n_taus)``.
    """

    taus = list(target_taus)
    cmd = _deepquantreg_command()
    output_dir.mkdir(parents=True, exist_ok=True)
    in_path = output_dir / "deepquantreg_input.npz"
    out_path = output_dir / "deepquantreg_predictions.npz"
    np.savez_compressed(
        in_path,
        X_train=dataset.X,
        y_obs=dataset.y_obs,
        y_true=dataset.y_true,
        delta=dataset.delta,
        L=dataset.L,
        R=dataset.R,
        X_test=X_test,
        taus=np.array(taus, dtype=float),
    )
    full_cmd = cmd + _deepquantreg_runner_options(include_defaults=True) + [str(in_path), str(out_path)]
    (output_dir / "deepquantreg_command.txt").write_text(
        shlex.join(full_cmd) + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        full_cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(message or f"DeepQuantreg command failed with exit code {proc.returncode}")
    pred = np.load(out_path)["pred"]
    pred = np.asarray(pred, dtype=float)
    expected_shape = (X_test.shape[0], len(taus))
    if pred.shape != expected_shape:
        raise ValueError(f"DeepQuantreg pred shape {pred.shape} != expected {expected_shape}")
    return pred


def run_deepquantreg_bootstrap_predictions(
    dataset: SimDataset,
    X_test: np.ndarray,
    target_taus: Iterable[float],
    output_dir: Path,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run paired-bootstrap DeepQuantreg predictions for PIW computation.

    The external command is invoked once per bootstrap sample. The raw inputs
    and predictions are kept under ``output_dir`` so every PIW value can be
    traced back to the primitive bootstrap predictions.
    """

    taus = list(target_taus)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = np.empty((int(B), X_test.shape[0], len(taus)), dtype=float)
    n = dataset.n
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        boot = SimDataset(
            X=dataset.X[idx],
            y_true=dataset.y_true[idx],
            y_obs=dataset.y_obs[idx],
            delta=dataset.delta[idx],
            L=dataset.L[idx],
            R=dataset.R[idx],
            scenario=dataset.scenario,
            error=dataset.error,
            censor_type=dataset.censor_type,
            censor_rate=dataset.censor_rate,
        )
        preds[b] = run_deepquantreg_adapter(
            boot,
            X_test,
            taus,
            output_dir / f"boot_{b:04d}",
        )
    return preds
