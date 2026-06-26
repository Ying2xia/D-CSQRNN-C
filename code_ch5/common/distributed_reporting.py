"""Progress reporting and incremental output helpers for Section 5.5."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import summarize
from .storage import save_csv


ACCURACY_VALUE_COLS = ["ql_ratio", "ree", "coverage", "benchmark_coverage"]
TIMING_VALUE_COLS = ["time_seconds", "ract"]
PAPER_TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
CENSOR_TYPE_ORDER = {"left": 0, "right": 1, "interval": 2}
METHOD_ORDER = {"DA-CSQRNN": 0, "DCS-QRNN-C": 1, "OS": 2, "One-shot": 2}


def _available_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def print_setting_summary(
    rows: list[dict[str, object]],
    label: str,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> None:
    """Print a compact mean table for one completed Section 5.5 setting."""

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no rows")
        return
    groups = _available_columns(df, group_cols)
    values = _available_columns(df, value_cols)
    if not groups or not values:
        print(f"\n[{label}] rows saved")
        return
    table = df.groupby(groups, dropna=False)[values].mean().reset_index()
    print(f"\n[{label}] mean metrics over {df.get('rep', pd.Series(dtype=object)).nunique()} replication(s)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_timing_summary(rows: list[dict[str, object]], label: str) -> None:
    """Print mean wall-clock time and RACT for one completed Section 5.5 setting."""

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no timing rows")
        return
    values = _available_columns(df, TIMING_VALUE_COLS)
    if not values:
        print(f"\n[{label}] timing rows saved")
        return
    groups = _available_columns(df, ["method", "K", "R"])
    table = df.groupby(groups, dropna=False)[values].mean().reset_index()
    if "time_seconds" in table.columns:
        table["time_minutes"] = table["time_seconds"] / 60.0
    if "K" in table.columns:
        table["K"] = table["K"].apply(lambda x: "" if pd.isna(x) else int(x))
    if "R" in table.columns:
        table["R"] = table["R"].apply(lambda x: "" if pd.isna(x) else int(x))
    ordered = [col for col in ["method", "K", "R", "time_seconds", "time_minutes", "ract"] if col in table.columns]
    table = table[ordered]
    print(f"\n[{label}] mean timing over {df.get('rep', pd.Series(dtype=object)).nunique()} replication(s)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_progress(
    run_dir: Path,
    metric_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]],
    timing_rows: list[dict[str, object]],
    metric_group_cols: Sequence[str] | None = None,
    metric_summary_name: str = "summary.csv",
    timing_group_cols: Sequence[str] | None = None,
    timing_summary_name: str = "summary_timing.csv",
) -> None:
    """Refresh raw CSVs and summaries after each completed setting."""

    if metric_rows:
        raw = save_csv(run_dir / "metrics_raw.csv", metric_rows)
        if metric_group_cols is not None:
            values = _available_columns(raw, ACCURACY_VALUE_COLS)
            save_csv(run_dir / metric_summary_name, summarize(raw, list(metric_group_cols), values))
    if fit_rows:
        save_csv(run_dir / "fit_log.csv", fit_rows)
    if timing_rows:
        raw_time = save_csv(run_dir / "timing_raw.csv", timing_rows)
        if timing_group_cols is not None:
            values = _available_columns(raw_time, TIMING_VALUE_COLS)
            save_csv(run_dir / timing_summary_name, summarize(raw_time, list(timing_group_cols), values))


def _paper_method_label(method: str, K) -> str:
    if method in {"DA-CSQRNN", "DA-CSQRNN-centralized"}:
        return "DA-CSQRNN"
    if method == "DCS-QRNN-C":
        return f"DCS-QRNN-C (K={int(K)})"
    if method in {"OS", "One-shot"}:
        return f"OS (K={int(K)})"
    return str(method)


def _paper_method_order(method: str, K) -> int:
    base = METHOD_ORDER.get(method, 99)
    return base * 1000 if pd.isna(K) else base * 1000 + int(K)


def save_accuracy_paper_tables(
    run_dir: Path,
    metric_rows: list[dict[str, object]],
    taus: Sequence[float] = PAPER_TAUS,
) -> None:
    """Write wide Section 5.5.2 tables matching the thesis display.

    The raw simulation output remains long-form.  These files pivot the mean
    values into one row per (scenario, error, metric block, censoring rate,
    censoring type, method), with one column for each reported tau level.
    """

    df = pd.DataFrame(metric_rows)
    if df.empty:
        return

    methods = {"DA-CSQRNN", "DA-CSQRNN-centralized", "DCS-QRNN-C", "OS", "One-shot"}
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        return

    df["paper_method"] = [
        _paper_method_label(method, K) for method, K in zip(df["method"], df["K"])
    ]
    df["method_order"] = [
        _paper_method_order(method, K) for method, K in zip(df["method"], df["K"])
    ]
    df["type_order"] = df["censor_type"].map(CENSOR_TYPE_ORDER).fillna(99).astype(int)
    df["rate_order"] = (df["censor_rate"].astype(float) * 100).round().astype(int)

    rows = []
    group_cols = [
        "scenario",
        "error",
        "censor_rate",
        "rate_order",
        "censor_type",
        "type_order",
        "paper_method",
        "method_order",
    ]
    for metric in ("ql_ratio", "ree"):
        if metric not in df.columns:
            continue
        sub = df.copy()
        if metric == "ree":
            sub = sub[~sub["method"].isin({"DA-CSQRNN", "DA-CSQRNN-centralized"})]
        sub = sub[np.isfinite(pd.to_numeric(sub[metric], errors="coerce"))]
        if sub.empty:
            continue
        mean = sub.groupby(group_cols + ["tau"], dropna=False)[metric].mean().reset_index()
        wide = (
            mean.pivot_table(index=group_cols, columns="tau", values=metric, aggfunc="mean")
            .reset_index()
        )
        wide.insert(0, "metric", metric)
        rows.append(wide)

    if not rows:
        return

    out = pd.concat(rows, ignore_index=True)
    for tau in taus:
        key = round(float(tau), 2)
        if key in out.columns:
            out = out.rename(columns={key: f"tau_{key:g}"})
        elif float(key) in out.columns:
            out = out.rename(columns={float(key): f"tau_{key:g}"})
        else:
            out[f"tau_{key:g}"] = np.nan

    tau_cols = [f"tau_{round(float(t), 2):g}" for t in taus]
    out = out.sort_values(
        ["scenario", "error", "metric", "rate_order", "type_order", "method_order"],
        kind="stable",
    )
    public_cols = [
        "scenario",
        "error",
        "metric",
        "censor_rate",
        "censor_type",
        "paper_method",
        *tau_cols,
    ]
    out_public = out[public_cols]

    table_dir = Path(run_dir) / "paper_tables"
    save_csv(table_dir / "distributed_accuracy_wide.csv", out_public)
    for (scenario, error), sub in out_public.groupby(["scenario", "error"], dropna=False):
        error_tag = "t" if str(error) == "t3" else str(error)
        save_csv(table_dir / f"distributed_accuracy_s{int(scenario)}_{error_tag}.csv", sub)


def save_timing_paper_table(run_dir: Path, timing_rows: list[dict[str, object]]) -> None:
    """Write the Section 5.5 timing table in the thesis row layout."""

    df = pd.DataFrame(timing_rows)
    if df.empty:
        return
    rows = []
    central = df[df["method"].isin(["DA-CSQRNN", "DA-CSQRNN-centralized"])]
    if not central.empty:
        rows.append(
            {
                "method": "DA-CSQRNN",
                "K": np.nan,
                "time_seconds": central["time_seconds"].mean(),
                "ract": 1.0,
            }
        )
    for method, label in [("DCS-QRNN-C", "DCS-QRNN-C"), ("OS", "OS"), ("One-shot", "OS")]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        mean = sub.groupby("K", dropna=False)[["time_seconds", "ract"]].mean().reset_index()
        for _, row in mean.iterrows():
            rows.append(
                {
                    "method": label,
                    "K": int(row["K"]),
                    "time_seconds": row["time_seconds"],
                    "ract": row["ract"],
                }
            )
    if rows:
        save_csv(Path(run_dir) / "paper_tables" / "distributed_timing.csv", rows)


def save_worker_paper_table(
    run_dir: Path,
    metric_rows: list[dict[str, object]],
    timing_rows: list[dict[str, object]],
    n_train: int,
    tau: float,
) -> None:
    """Write the Section 5.5 worker-sensitivity table."""

    metrics = pd.DataFrame(metric_rows)
    timings = pd.DataFrame(timing_rows)
    if metrics.empty:
        return
    metrics = metrics[(metrics["method"] == "DCS-QRNN-C") & (metrics["tau"].astype(float).round(8) == round(float(tau), 8))]
    if metrics.empty:
        return
    out = metrics.groupby("K", dropna=False)[["ql_ratio", "ree"]].mean().reset_index()
    if not timings.empty:
        timing = timings[timings["method"] == "DCS-QRNN-C"].groupby("K", dropna=False)[["ract"]].mean().reset_index()
        out = out.merge(timing, on="K", how="left")
    out["K"] = out["K"].astype(int)
    out.insert(1, "N_k", (int(n_train) // out["K"]).astype(int))
    save_csv(Path(run_dir) / "paper_tables" / "distributed_workers.csv", out.sort_values("K"))
