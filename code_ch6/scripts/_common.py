"""Shared CLI helpers for Chapter 6 real-data scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=str(ROOT / "results"), help="Directory for Chapter 6 outputs.")
    parser.add_argument("--seed", type=int, default=20260508, help="Base seed for splits, censoring levels, DA draws, and model starts.")
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run with fewer rows, reps, DA iterations, and optimizer steps.")
    parser.add_argument("--n-rep", type=int, default=None, help="Number of random train/test splits or repetitions.")
    parser.add_argument("--max-iter", type=int, default=None, help="Adam optimization steps for each QRNN fit.")
    parser.add_argument("--J", type=int, default=10, help="Hidden-layer node count.")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01, help="Regularization strength.")
    parser.add_argument("--hyperparams", default=None, help="CSV from run_6_0_hyperparameters.py with per-tau J and lambda values.")
    parser.add_argument("--backend", choices=["numpy", "torch", "auto"], default="auto", help="Model-fitting backend.")
    parser.add_argument("--device", default="auto", help="Device for PyTorch backend: cpu, cuda, cuda:0, mps, or auto.")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32", help="PyTorch floating-point dtype.")


def apply_runtime_defaults(args, *, n_rep: int, max_iter: int, quick_max_iter: int = 35):
    if args.quick:
        args.n_rep = args.n_rep or 1
        args.max_iter = args.max_iter or quick_max_iter
    else:
        args.n_rep = args.n_rep or n_rep
        args.max_iter = args.max_iter or max_iter
    print_runtime_device(args)
    return args


def print_runtime_device(args) -> None:
    backend = getattr(args, "backend", "numpy")
    device = getattr(args, "device", "cpu")
    torch_dtype = getattr(args, "torch_dtype", "float32")
    try:
        from common.torch_backend import resolve_device

        resolved_backend, resolved_device = resolve_device(device=device, backend=backend)
    except Exception as exc:
        print(
            f"[runtime] backend={backend}, device={device}, torch_dtype={torch_dtype}; "
            f"device check failed: {exc}"
        )
        return

    if resolved_backend == "torch":
        label = "GPU" if str(resolved_device).startswith(("cuda", "mps")) else "CPU"
        detail = ""
        if str(resolved_device).startswith("cuda"):
            try:
                import torch

                idx = torch.cuda.current_device()
                detail = f", name={torch.cuda.get_device_name(idx)}"
            except Exception:
                detail = ""
        print(
            f"[runtime] using {label}: backend=torch, device={resolved_device}, "
            f"torch_dtype={torch_dtype}{detail}"
        )
    else:
        print("[runtime] using CPU: backend=numpy, device=cpu")
