#!/usr/bin/env python3
"""Plot Section 5.4.4 comparison figures from result CSV files.

Each simulation setting gets two PNG figures:

1. QL_ratio line plot across target taus.
2. PIW_ratio line plot with Coverage_C / Coverage_U grouped bars.

The script reads existing result folders. It plots DeepQuantreg automatically
when valid DeepQuantreg rows are present, and otherwise omits it.
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
BAR_COLORS = {
    ("DA-CSQRNN", "C"): "#93c5fd",
    ("DA-CSQRNN", "U"): "#1d4ed8",
    ("DAqrnn", "C"): "#fca5a5",
    ("DAqrnn", "U"): "#b91c1c",
    ("DeepQuantreg", "C"): "#86efac",
    ("DeepQuantreg", "U"): "#047857",
}
CENSOR_LABELS = {"left": "Left", "right": "Right", "interval": "Interval"}
ERROR_LABELS = {"normal": "N(0, 1)", "t3": "t(3)"}


@dataclass(frozen=True)
class Setting:
    scenario: int
    error: str
    censor_type: str
    censor_rate: float

    @property
    def tag(self) -> str:
        rate = int(round(self.censor_rate * 100))
        return f"s{self.scenario}_{self.error}_{self.censor_type}_{rate}"

    @property
    def title(self) -> str:
        rate = int(round(self.censor_rate * 100))
        return (
            f"Scenario {self.scenario} | {ERROR_LABELS.get(self.error, self.error)} | "
            f"{CENSOR_LABELS.get(self.censor_type, self.censor_type)} censoring | {rate}%"
        )


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
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    required = {"scenario", "error", "censor_type", "censor_rate", "method", "tau"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
    return df


def normalise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["method"] = out["method"].replace({"DACSQRNN": "DA-CSQRNN", "DAQRNN": "DAqrnn"})
    if "coverage_c" not in out.columns:
        out["coverage_c"] = out.get("coverage", np.nan)
    if "coverage_u" not in out.columns:
        out["coverage_u"] = out.get("benchmark_coverage", np.nan)
    for col in ["scenario", "censor_rate", "tau", "ql_ratio", "piw_ratio", "coverage_c", "coverage_u"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_combined_metrics(comparison_path: Path, dacsqrnn_path: Path | None) -> pd.DataFrame:
    comparison = normalise_metrics(read_metrics(comparison_path))
    frames = [comparison] if not comparison.empty else []

    if dacsqrnn_path is not None:
        dacs = normalise_metrics(read_metrics(dacsqrnn_path))
        if not dacs.empty:
            comparison_methods = set(comparison["method"].dropna().astype(str)) if not comparison.empty else set()
            if "DA-CSQRNN" not in comparison_methods:
                frames.append(dacs[dacs["method"] == "DA-CSQRNN"].copy())

    if not frames:
        raise FileNotFoundError("No metric rows found in the requested result files.")

    combined = pd.concat(frames, ignore_index=True)
    keep_cols = [
        "scenario",
        "error",
        "censor_type",
        "censor_rate",
        "method",
        "tau",
        "ql_ratio",
        "piw_ratio",
        "coverage_c",
        "coverage_u",
    ]
    combined = combined[[col for col in keep_cols if col in combined.columns]].copy()
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()
    return combined


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "error", "censor_type", "censor_rate", "method", "tau"]
    value_cols = ["ql_ratio", "piw_ratio", "coverage_c", "coverage_u"]
    return (
        df.groupby(group_cols, dropna=False)[value_cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )


def filter_settings(settings: list[Setting], args: argparse.Namespace) -> list[Setting]:
    out = settings
    if args.scenario is not None:
        out = [s for s in out if s.scenario == args.scenario]
    if args.error is not None:
        out = [s for s in out if s.error == args.error]
    if args.censor_type is not None:
        out = [s for s in out if s.censor_type == args.censor_type]
    if args.censor_rate is not None:
        out = [s for s in out if float(s.censor_rate) == float(args.censor_rate)]
    return out


def discover_settings(summary: pd.DataFrame, args: argparse.Namespace) -> list[Setting]:
    rows = summary[["scenario", "error", "censor_type", "censor_rate"]].drop_duplicates()
    settings = [
        Setting(int(row.scenario), str(row.error), str(row.censor_type), float(row.censor_rate))
        for row in rows.itertuples(index=False)
    ]
    settings = sorted(settings, key=lambda s: (s.scenario, s.error, s.censor_type, s.censor_rate))
    settings = filter_settings(settings, args)
    if not settings:
        raise ValueError("No settings match the requested filters.")
    return settings


def setting_data(summary: pd.DataFrame, setting: Setting) -> pd.DataFrame:
    return summary[
        (summary["scenario"].astype(int) == setting.scenario)
        & (summary["error"].astype(str) == setting.error)
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


def plot_ql(plt, setting: Setting, data: pd.DataFrame, output: Path, dpi: int) -> None:
    methods = available_methods(data, ["ql_ratio"])
    if not methods:
        return

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


def coverage_bar_series(data: pd.DataFrame, methods: list[str]) -> list[tuple[str, str, str, str]]:
    series: list[tuple[str, str, str, str]] = []
    for method in methods:
        for kind, col in (("C", "coverage_c"), ("U", "coverage_u")):
            sub = data[data["method"] == method]
            if col in sub.columns and np.isfinite(pd.to_numeric(sub[col], errors="coerce")).any():
                label = f"{METHOD_LABELS[method]} Coverage$_{kind}$"
                series.append((method, kind, col, label))
    return series


def plot_piw_coverage(plt, setting: Setting, data: pd.DataFrame, output: Path, dpi: int) -> None:
    methods = available_methods(data, ["piw_ratio", "coverage_c", "coverage_u"])
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=dpi)
    ax2 = ax.twinx()
    configure_axis(ax, "PIW_ratio")
    ax2.set_ylabel("Coverage", fontsize=11)
    ax2.spines["top"].set_visible(False)

    all_piw: list[float] = []
    line_handles = []
    line_labels = []
    for method in methods:
        x, y = method_series(data, method, "piw_ratio")
        if y.size == 0:
            continue
        all_piw.extend(y.tolist())
        handle = ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            linestyle=LINE_STYLES[method],
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} PIW",
            zorder=4,
        )[0]
        line_handles.append(handle)
        line_labels.append(f"{METHOD_LABELS[method]} PIW")

    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_ylim(*padded_ylim(all_piw, include=[1.0], lower_bound=0.0))

    bar_info = coverage_bar_series(data, methods)
    bar_handles = []
    bar_labels = []
    if bar_info:
        n_bars = len(bar_info)
        group_width = 0.42 if n_bars <= 4 else 0.66
        bar_width = min(0.12, group_width / max(n_bars, 1) * 0.82)
        offsets = np.linspace(
            -group_width / 2.0 + bar_width / 2.0,
            group_width / 2.0 - bar_width / 2.0,
            n_bars,
        )
        for idx, (method, kind, col, label) in enumerate(bar_info):
            heights: list[float] = []
            for tau in TAUS:
                row = data[(data["method"] == method) & np.isclose(data["tau"].astype(float), tau)]
                heights.append(float(row.iloc[0][col]) if not row.empty and np.isfinite(row.iloc[0][col]) else np.nan)
            bars = ax2.bar(
                TAU_POSITIONS + offsets[idx],
                heights,
                width=bar_width,
                color=BAR_COLORS[(method, kind)],
                edgecolor="#374151",
                linewidth=0.35,
                alpha=0.78,
                label=label,
                zorder=1,
            )
            bar_handles.append(bars[0])
            bar_labels.append(label)
    ax2.set_ylim(0.0, 1.0)
    ax2.axhline(0.95, color="#9ca3af", linestyle=":", linewidth=1.2, zorder=0)

    handles = line_handles + bar_handles
    labels = line_labels + bar_labels
    ncol = 2 if len(labels) <= 4 else 3
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        frameon=False,
        fontsize=7.4,
        columnspacing=1.2,
        handlelength=2.2,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    ensure_dir(output.parent)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-results",
        default=str(ROOT / "results" / "5_4_4_daqrnn_all_censoring"),
        help="Directory or metrics_raw.csv containing DAqrnn/DeepQuantreg comparison rows.",
    )
    parser.add_argument(
        "--dacsqrnn-results",
        default=str(ROOT / "results" / "5_4_2_ql_piw"),
        help="Directory or metrics_raw.csv containing DA-CSQRNN rows to merge when absent from comparison results.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory for figures. Default: <comparison-results>/comparison_figures.")
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=None)
    parser.add_argument("--error", choices=["normal", "t3"], default=None)
    parser.add_argument("--censor-type", choices=["left", "right", "interval"], default=None)
    parser.add_argument("--censor-rate", type=float, choices=[0.25, 0.50], default=None)
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison_results)
    dacs_path = Path(args.dacsqrnn_results) if args.dacsqrnn_results else None
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else resolve_metrics_path(comparison_path).parent / "comparison_figures"
    )
    out_dir = ensure_dir(out_dir)

    combined = load_combined_metrics(comparison_path, dacs_path)
    summary = summarise_metrics(combined)
    settings = discover_settings(summary, args)
    save_csv(out_dir / "comparison_plot_data.csv", summary)

    plt = load_matplotlib()
    manifest: list[dict[str, object]] = []
    for setting in settings:
        sub = setting_data(summary, setting)
        methods = ",".join(available_methods(sub, ["ql_ratio", "piw_ratio", "coverage_c", "coverage_u"]))
        ql_path = out_dir / f"{setting.tag}_ql_ratio.png"
        piw_path = out_dir / f"{setting.tag}_piw_coverage.png"
        plot_ql(plt, setting, sub, ql_path, args.dpi)
        plot_piw_coverage(plt, setting, sub, piw_path, args.dpi)
        manifest.append(
            {
                "scenario": setting.scenario,
                "error": setting.error,
                "censor_type": setting.censor_type,
                "censor_rate": setting.censor_rate,
                "methods": methods,
                "ql_ratio_figure": str(ql_path),
                "piw_coverage_figure": str(piw_path),
            }
        )

    save_csv(out_dir / "manifest.csv", manifest)
    print(f"saved {len(manifest) * 2} figures to {out_dir}")


if __name__ == "__main__":
    main()
