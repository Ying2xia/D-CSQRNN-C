"""Filesystem helpers for raw and summary outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(
    base_dir: str | Path,
    section: str,
    run_name: str | None = None,
    clean: bool = True,
) -> Path:
    """Create the output directory for one section run.

    Parameters
    ----------
    base_dir
        Root results directory, normally ``results``.
    section
        Section-specific subdirectory such as ``"5_4_2_ql_piw"``.
    run_name
        Optional nested run name when one section needs multiple independent
        output folders.
    clean
        If true, remove the section/run directory before recreating it. This
        prevents stale raw files from an older run, for example when rerunning
        with a smaller ``S`` or fewer replications.
    """
    root = ensure_dir(Path(base_dir).expanduser().resolve())
    run_dir = root / section / run_name if run_name else root / section
    if clean and run_dir.exists():
        shutil.rmtree(run_dir)
    return ensure_dir(run_dir)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    path = Path(path)
    ensure_dir(path.parent)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
