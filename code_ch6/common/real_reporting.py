"""Reporting helpers for Chapter 6 real-data outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import summarize
from .storage import save_csv


TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
CENSOR_ORDER = {"left": 0, "right": 1, "interval": 2}
METHOD_ORDER = {"DA-CSQRNN": 0, "DAqrnn": 1, "DeepQuantreg": 2, "DCS-QRNN-C": 1, "OS": 2}


def _tau_cols(taus: Sequence[float] = TAUS) -> list[str]:
    return [f"tau_{float(t):g}" for t in taus]


def print_metric_summary(rows: list[dict[str, object]], label: str, group_cols: Sequence[str], value_cols: Sequence[str]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n[{label}] no rows")
        return
    cols = [col for col in value_cols if col in df.columns]
    groups = [col for col in group_cols if col in df.columns]
    table = df.groupby(groups, dropna=False)[cols].mean().reset_index()
    print(f"\n[{label}] mean over {df.get('rep', pd.Series(dtype=object)).nunique()} replication(s)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_timing_summary(rows: list[dict[str, object]], label: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    group_cols = [col for col in ["method", "K", "R"] if col in df.columns]
    table = df.groupby(group_cols, dropna=False)[["time_seconds", "ract"]].mean().reset_index()
    table["time_minutes"] = table["time_seconds"] / 60.0
    for col in ["K", "R"]:
        if col in table.columns:
            table[col] = table[col].apply(lambda x: "" if pd.isna(x) else int(x))
    order = [col for col in ["method", "K", "R", "time_seconds", "time_minutes", "ract"] if col in table.columns]
    print(f"\n[{label}] timing")
    print(table[order].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_progress(
    run_dir: Path,
    metric_rows: list[dict[str, object]],
    fit_rows: list[dict[str, object]] | None = None,
    timing_rows: list[dict[str, object]] | None = None,
    censor_rows: list[dict[str, object]] | None = None,
    metric_group_cols: Sequence[str] | None = None,
    metric_value_cols: Sequence[str] = ("ql_ratio", "piw_ratio", "ree"),
) -> None:
    if metric_rows:
        raw = save_csv(run_dir / "metrics_raw.csv", metric_rows)
        if metric_group_cols is not None:
            values = [col for col in metric_value_cols if col in raw.columns]
            save_csv(run_dir / "summary.csv", summarize(raw, list(metric_group_cols), values))
    if fit_rows:
        save_csv(run_dir / "fit_log.csv", fit_rows)
    if timing_rows:
        timing = save_csv(run_dir / "timing_raw.csv", timing_rows)
        save_csv(run_dir / "summary_timing.csv", summarize(timing, ["dataset", "censor_type", "censor_rate", "method", "K", "R"], ["time_seconds", "ract"]))
    if censor_rows:
        save_csv(run_dir / "censoring_summary.csv", censor_rows)


def _wide_by_tau(
    df: pd.DataFrame,
    index_cols: list[str],
    metric: str,
    taus: Sequence[float] = TAUS,
) -> pd.DataFrame:
    mean = df.groupby(index_cols + ["tau"], dropna=False)[metric].mean().reset_index()
    wide = mean.pivot_table(index=index_cols, columns="tau", values=metric, aggfunc="mean").reset_index()
    for tau in taus:
        key = round(float(tau), 2)
        if key in wide.columns:
            wide = wide.rename(columns={key: f"tau_{key:g}"})
        elif float(key) in wide.columns:
            wide = wide.rename(columns={float(key): f"tau_{key:g}"})
        else:
            wide[f"tau_{key:g}"] = np.nan
    return wide[index_cols + _tau_cols(taus)]


def save_central_main_tables(run_dir: Path, metric_rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return
    out = []
    for metric in ["ql_ratio", "piw_ratio"]:
        sub = df[np.isfinite(pd.to_numeric(df[metric], errors="coerce"))].copy()
        if sub.empty:
            continue
        wide = _wide_by_tau(sub, ["dataset", "censor_rate", "censor_type"], metric)
        wide.insert(0, "metric", metric)
        out.append(wide)
    if not out:
        return
    table = pd.concat(out, ignore_index=True)
    table["type_order"] = table["censor_type"].map(CENSOR_ORDER).fillna(99).astype(int)
    table = table.sort_values(["dataset", "metric", "censor_rate", "type_order"], kind="stable").drop(columns=["type_order"])
    table_dir = Path(run_dir) / "paper_tables"
    save_csv(table_dir / "centralized_real_main_wide.csv", table)
    for dataset, sub in table.groupby("dataset", dropna=False):
        save_csv(table_dir / f"{dataset}_main_wide.csv", sub)


def save_central_comparison_tables(run_dir: Path, metric_rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return
    df = df.copy()
    df["type_order"] = df["censor_type"].map(CENSOR_ORDER).fillna(99).astype(int)
    df["method_order"] = df["method"].map(METHOD_ORDER).fillna(99).astype(int)
    wide = _wide_by_tau(df, ["dataset", "censor_rate", "censor_type", "type_order", "method", "method_order"], "ql_ratio")
    wide = wide.sort_values(["dataset", "censor_rate", "type_order", "method_order"], kind="stable")
    wide = wide.drop(columns=["type_order", "method_order"])
    table_dir = Path(run_dir) / "paper_tables"
    save_csv(table_dir / "centralized_real_comparison_wide.csv", wide)
    for dataset, sub in wide.groupby("dataset", dropna=False):
        save_csv(table_dir / f"{dataset}_comparison_wide.csv", sub)


def save_distributed_tables(run_dir: Path, metric_rows: list[dict[str, object]], timing_rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return
    df = df.copy()
    df["K_table"] = df["K"].apply(lambda x: "--" if pd.isna(x) else str(int(x)))
    df["type_order"] = df["censor_type"].map(CENSOR_ORDER).fillna(99).astype(int)
    df["method_order"] = df["method"].map(METHOD_ORDER).fillna(99).astype(int)
    ql = _wide_by_tau(df, ["dataset", "censor_rate", "censor_type", "type_order", "method", "method_order", "K_table"], "ql_ratio")
    ql.insert(0, "metric", "ql_ratio")
    ree_df = df[df["method"] != "DA-CSQRNN"].copy()
    ree = _wide_by_tau(ree_df, ["dataset", "censor_rate", "censor_type", "type_order", "method", "method_order", "K_table"], "ree")
    ree.insert(0, "metric", "ree")
    out = pd.concat([ql, ree], ignore_index=True)
    out = out.rename(columns={"K_table": "K"})
    out = out.sort_values(["metric", "censor_rate", "type_order", "method_order", "K"], kind="stable")
    out = out.drop(columns=["type_order", "method_order"])
    table_dir = Path(run_dir) / "paper_tables"
    save_csv(table_dir / "distributed_real_ql_ree_wide.csv", out)

    if timing_rows:
        timing = pd.DataFrame(timing_rows)
        timing_mean = timing.groupby(["dataset", "censor_rate", "censor_type", "method", "K", "R"], dropna=False)[["time_seconds", "ract"]].mean().reset_index()
        save_csv(table_dir / "distributed_real_timing.csv", timing_mean)
