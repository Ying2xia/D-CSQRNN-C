#!/usr/bin/env python3
"""Plot Section 6.1.4 real-data comparison QL_ratio figures.

DeepQuantreg is plotted automatically when valid rows are present, and omitted
otherwise.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _common import ROOT
from common.storage import ensure_dir, save_csv


TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
TAU_POSITIONS = np.arange(len(TAUS), dtype=float)
TAU_LABELS = [f"{tau:g}" for tau in TAUS]
METHOD_ORDER = ("DA-CSQRNN", "DAqrnn", "DeepQuantreg")
METHOD_LABELS = {
    "DA-CSQRNN": "DA-CSQRNN",
    "DAqrnn": "DAqrnn",
    "DeepQuantreg": "DeepQuantreg",
}
METHOD_COLORS = {
    "DA-CSQRNN": "#2563eb",
    "DAqrnn": "#dc2626",
    "DeepQuantreg": "#059669",
}
LINE_STYLES = {
    "DA-CSQRNN": "-",
    "DAqrnn": "--",
    "DeepQuantreg": "-.",
}

@dataclass(frozen=True)
class Setting:
    dataset: str
    censor_type: str
    censor_rate: float

    @property
    def tag(self) -> str:
        rate = int(round(self.censor_rate * 100))
        return f"{self.dataset}_{self.censor_type}_{rate}"


def load_matplotlib():
    try:
        cache_dir = ensure_dir(ROOT / ".matplotlib-cache")
        xdg_cache_dir = ensure_dir(ROOT / ".cache")
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to draw these PNG figures. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return plt


def resolve_metrics_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "metrics_raw.csv" if path.is_dir() else path


def read_metrics(path: Path) -> pd.DataFrame:
    metrics_path = resolve_metrics_path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    required = {"dataset", "censor_type", "censor_rate", "method", "tau"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
    return df


def _from_long_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accept summary.csv files with metric/value_mean columns as input."""

    if not {"metric", "value_mean"}.issubset(df.columns):
        return df
    index_cols = ["dataset", "censor_type", "censor_rate", "method", "tau"]
    wide = (
        df.pivot_table(index=index_cols, columns="metric", values="value_mean", aggfunc="mean")
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def normalise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _from_long_summary(df).copy()
    out["method"] = out["method"].replace({"DACSQRNN": "DA-CSQRNN", "DAQRNN": "DAqrnn"})
    for col in ["censor_rate", "tau", "ql_ratio"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep_cols = ["dataset", "censor_type", "censor_rate", "method", "tau", "ql_ratio"]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep_cols].copy()
    return out[out["method"].isin(METHOD_ORDER)].copy()


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "censor_type", "censor_rate", "method", "tau"]
    value_cols = ["ql_ratio"]
    return (
        df.groupby(group_cols, dropna=False)[value_cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )


def filter_settings(settings: list[Setting], args: argparse.Namespace) -> list[Setting]:
    out = settings
    if args.dataset is not None:
        out = [s for s in out if s.dataset == args.dataset]
    if args.censor_type is not None:
        out = [s for s in out if s.censor_type == args.censor_type]
    if args.censor_rate is not None:
        out = [s for s in out if float(s.censor_rate) == float(args.censor_rate)]
    return out


def discover_settings(summary: pd.DataFrame, args: argparse.Namespace) -> list[Setting]:
    rows = summary[["dataset", "censor_type", "censor_rate"]].drop_duplicates()
    settings = [
        Setting(str(row.dataset), str(row.censor_type), float(row.censor_rate))
        for row in rows.itertuples(index=False)
    ]
    settings = sorted(settings, key=lambda s: (s.dataset, s.censor_type, s.censor_rate))
    settings = filter_settings(settings, args)
    if not settings:
        raise ValueError("No settings match the requested filters.")
    return settings


def setting_data(summary: pd.DataFrame, setting: Setting) -> pd.DataFrame:
    return summary[
        (summary["dataset"].astype(str) == setting.dataset)
        & (summary["censor_type"].astype(str) == setting.censor_type)
        & np.isclose(summary["censor_rate"].astype(float), setting.censor_rate)
    ].copy()


def available_methods(df: pd.DataFrame, metric_cols: Iterable[str]) -> list[str]:
    methods: list[str] = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        if any(col in sub.columns and np.isfinite(pd.to_numeric(sub[col], errors="coerce")).any() for col in metric_cols):
            methods.append(method)
    return methods


def method_series(data: pd.DataFrame, method: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    sub = data[data["method"] == method].copy()
    xs: list[float] = []
    ys: list[float] = []
    for idx, tau in enumerate(TAUS):
        row = sub[np.isclose(sub["tau"].astype(float), tau)]
        if row.empty:
            continue
        value = float(row.iloc[0][value_col])
        if np.isfinite(value):
            xs.append(float(idx))
            ys.append(value)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def padded_ylim(values: Iterable[float], include: Iterable[float] = (), lower_bound: float | None = None) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    extra = np.asarray([v for v in include if np.isfinite(v)], dtype=float)
    if extra.size:
        arr = np.concatenate([arr, extra])
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if lo == hi:
        pad = max(abs(lo) * 0.08, 0.05)
    else:
        pad = (hi - lo) * 0.12
    lo -= pad
    hi += pad
    if lower_bound is not None:
        lo = max(lower_bound, lo)
    return lo, hi


def configure_axis(ax, ylabel: str) -> None:
    ax.set_xlabel(r"$\tau$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(TAU_POSITIONS)
    ax.set_xticklabels(TAU_LABELS)
    ax.set_xlim(-0.5, len(TAUS) - 0.5)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ql(plt, data: pd.DataFrame, output: Path, dpi: int) -> bool:
    methods = available_methods(data, ["ql_ratio"])
    if not methods:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=dpi)
    configure_axis(ax, "QL_ratio")
    all_values: list[float] = []
    for method in methods:
        x, y = method_series(data, method, "ql_ratio")
        if y.size == 0:
            continue
        all_values.extend(y.tolist())
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            linestyle=LINE_STYLES[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(*padded_ylim(all_values, include=[1.0], lower_bound=0.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(methods), frameon=False, fontsize=9)
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-results",
        default=str(ROOT / "results" / "6_1_4_deepquantreg_placeholder"),
        help="Directory or metrics_raw.csv containing Chapter 6.1.4 comparison rows.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory for figures. Default: <comparison-results>/comparison_figures_ql.")
    parser.add_argument("--dataset", choices=["boston", "gilgais"], default=None)
    parser.add_argument("--censor-type", choices=["left", "right", "interval"], default=None)
    parser.add_argument("--censor-rate", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison_results)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else resolve_metrics_path(comparison_path).parent / "comparison_figures_ql"
    )
    out_dir = ensure_dir(out_dir)

    combined = normalise_metrics(read_metrics(comparison_path))
    if combined.empty:
        raise FileNotFoundError("No plottable metric rows found in the requested result file.")
    summary = summarise_metrics(combined)
    settings = discover_settings(summary, args)
    save_csv(out_dir / "comparison_plot_data.csv", summary)

    plt = load_matplotlib()
    manifest: list[dict[str, object]] = []
    n_figures = 0
    for setting in settings:
        sub = setting_data(summary, setting)
        methods = ",".join(available_methods(sub, ["ql_ratio"]))
        ql_path = out_dir / f"{setting.tag}_ql_ratio.png"
        ql_saved = plot_ql(plt, sub, ql_path, args.dpi)
        n_figures += int(ql_saved)
        manifest.append(
            {
                "dataset": setting.dataset,
                "censor_type": setting.censor_type,
                "censor_rate": setting.censor_rate,
                "methods": methods,
                "ql_ratio_figure": str(ql_path) if ql_saved else "",
            }
        )

    save_csv(out_dir / "manifest.csv", manifest)
    print(f"saved {n_figures} figures to {out_dir}")


if __name__ == "__main__":
    main()
