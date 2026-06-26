"""Shared CLI helpers for section scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=str(ROOT / "results"), help="Directory for raw and summary results.")
    parser.add_argument("--seed", type=int, default=20260430, help="Base seed for training data, DA draws, bootstrap samples, and model starts.")
    parser.add_argument("--test-seed", type=int, default=730, help="Seed for the fixed uncensored test set shared across replicates.")
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run with reduced n, iterations, reps, and bootstrap samples.")
    parser.add_argument("--n-rep", type=int, default=None, help="Number of Monte Carlo replications. Default comes from the section config.")
    parser.add_argument("--n-train", type=int, default=None, help="Training sample size per replication.")
    parser.add_argument("--n-test", type=int, default=None, help="Uncensored test sample size per replication.")
    parser.add_argument("--max-iter", type=int, default=None, help="Adam optimization steps for each QRNN fit.")
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=None,
        help="Minimum benchmark PI sample count. Omit or set 0 to match the DA iteration count S; this does not add extra DA bootstraps.",
    )
    parser.add_argument("--J", type=int, default=10, help="Fallback hidden-layer node count when no hyperparameter CSV is supplied.")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01, help="Fallback regularization strength when no hyperparameter CSV is supplied.")
    parser.add_argument("--hyperparams", default=None, help="CSV from run_5_4_1_hyperparameters.py with per-tau J and lambda values.")
    parser.add_argument("--backend", choices=["numpy", "torch", "auto"], default="auto", help="Model-fitting backend: NumPy CPU, PyTorch, or automatic selection.")
    parser.add_argument("--device", default="auto", help="Device for PyTorch backend: cpu, cuda, cuda:0, mps, or auto.")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32", help="Floating-point dtype used by the PyTorch backend.")


def apply_quick_defaults(args, n_train_default, n_test_default, n_rep_default):
    if args.quick:
        args.n_train = args.n_train or min(100, n_train_default)
        args.n_test = args.n_test or min(100, n_test_default)
        args.n_rep = args.n_rep or 1
        args.max_iter = args.max_iter or 35
        args.bootstrap_reps = 0 if args.bootstrap_reps is None else args.bootstrap_reps
        args.S = getattr(args, "S", None) or 2
    else:
        args.n_train = args.n_train or n_train_default
        args.n_test = args.n_test or n_test_default
        args.n_rep = args.n_rep or n_rep_default
        args.max_iter = args.max_iter or 250
        args.bootstrap_reps = 0 if args.bootstrap_reps is None else args.bootstrap_reps
    print_runtime_device(args)
    return args


def print_runtime_device(args) -> None:
    """Print whether this run will use CPU NumPy or a PyTorch device."""

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
        print(f"[runtime] using CPU: backend=numpy, device=cpu")


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None, help="Run only one simulation scenario.")
    parser.add_argument("--error", choices=["normal", "t3"], default=None, help="Run only one error distribution.")
    parser.add_argument("--censor-type", choices=["left", "right", "interval"], default=None, help="Run only one censoring mechanism.")
    parser.add_argument("--censor-rate", type=float, choices=[0.25, 0.50], default=None, help="Run only one nominal censoring rate.")


def selected_settings(settings, args):
    for scenario, error, censor_type, censor_rate in settings:
        if getattr(args, "scenario", None) is not None and scenario != args.scenario:
            continue
        if getattr(args, "error", None) is not None and error != args.error:
            continue
        if getattr(args, "censor_type", None) is not None and censor_type != args.censor_type:
            continue
        if getattr(args, "censor_rate", None) is not None and float(censor_rate) != float(args.censor_rate):
            continue
        yield scenario, error, censor_type, censor_rate
